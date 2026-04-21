from text_rich_mllm.datasets import build_dataset_adapter


def test_docvqa_adapter_uses_hf_fields() -> None:
    adapter = build_dataset_adapter("docvqa")
    sample = adapter.convert_record(
        {
            "question_id": 7,
            "question": "Invoice number?",
            "answers": ["12345"],
            "image": "doc.png",
            "ocr_results": {"tokens": ["12345"]},
            "other_metadata": {"doc_id": "d1", "page_id": "p2"},
        },
        index=0,
        split="train",
    )
    assert sample.sample_id == "7"
    assert sample.question == "Invoice number?"
    assert sample.gold_answer == "12345"
    assert sample.metadata["doc_id"] == "d1"


def test_docvqa_adapter_hub_wds_nested_json_png() -> None:
    """pixparse/docvqa-wds：字段在 json 列，图像路径在 png。"""
    adapter = build_dataset_adapter("docvqa")
    sample = adapter.convert_record(
        {
            "json": {
                "questionId": "q42",
                "question": "Total due?",
                "answers": ["99.00"],
                "page_id": "pg1",
            },
            "png": "/cache/docvqa/pages/x.png",
        },
        index=0,
        split="train",
        image_root="/root/images",
    )
    assert sample.sample_id == "q42"
    assert sample.question == "Total due?"
    assert sample.gold_answer == "99.00"
    assert sample.image_path.endswith("x.png")
    assert sample.metadata["docvqa_json_nested"] is True


def test_chartqa_adapter_maps_query_and_label() -> None:
    adapter = build_dataset_adapter("chartqa")
    sample = adapter.convert_record(
        {
            "query": "What is the average?",
            "label": ["58"],
            "image": "chart.png",
            "human_or_machine": "human",
        },
        index=0,
        split="train",
    )
    assert sample.question == "What is the average?"
    assert sample.gold_answer == "58"
    assert sample.answer_type == "numeric"
    assert sample.metadata["source"] == "human"


def test_scienceqa_adapter_parses_option_string_and_maps_answer_to_letter() -> None:
    adapter = build_dataset_adapter("scienceqa")
    sample = adapter.convert_record(
        {
            "question": "[QUESTION]Which continent is highlighted?",
            "choices": "[OPTIONS](A) Europe (B) Antarctica (C) North America (D) Africa",
            "answer": "North America",
            "solution": "This continent is North America.",
            "CTH": False,
            "image": "map.png",
        },
        index=0,
        split="validation",
    )
    assert sample.question == "Which continent is highlighted?"
    assert sample.choices == ["Europe", "Antarctica", "North America", "Africa"]
    assert sample.gold_answer == "C"
    assert sample.metadata["solution"] == "This continent is North America."


def test_textvqa_adapter_maps_question_and_answers() -> None:
    adapter = build_dataset_adapter("textvqa")
    sample = adapter.convert_record(
        {
            "question_id": 42,
            "question": "What brand is shown?",
            "answers": ["nike", "nike inc", "nike"],
            "image": "scene.jpg",
        },
        index=0,
        split="train",
    )
    assert sample.sample_id == "42"
    assert sample.question == "What brand is shown?"
    assert sample.gold_answer == "nike"
    assert sample.task_type == "scene_text_qa"


def test_infographicvqa_adapter_uses_hub_question_id_field() -> None:
    """Ryoo72/InfographicsVQA 等快照使用 questionId（驼峰）作为问题主键。"""
    adapter = build_dataset_adapter("infographicvqa")
    sample = adapter.convert_record(
        {"questionId": "q-99", "question": "Units?", "answers": ["million"], "image": "info.png"},
        index=0,
        split="train",
    )
    assert sample.sample_id == "q-99"
    assert sample.gold_answer == "million"


def test_infographicvqa_adapter_flat_record() -> None:
    adapter = build_dataset_adapter("infographicvqa")
    sample = adapter.convert_record(
        {
            "sample_id": "iv-1",
            "question": "What is the largest segment?",
            "ground_truth": "Services",
            "image": "info.png",
        },
        index=0,
        split="validation",
    )
    assert sample.sample_id == "iv-1"
    assert sample.gold_answer == "Services"
    assert sample.task_type == "infographic_qa"


def test_infographicvqa_adapter_ignores_non_due_annotations_lists() -> None:
    """HF 行若含 name + 非 DUE 的 annotations，不得误判为 document.jsonl 导致 0 条样本。"""
    adapter = build_dataset_adapter("infographicvqa")
    samples = adapter.convert_records(
        [
            {
                "name": "noise",
                "annotations": [{"bbox": [0, 0, 1, 1]}],
                "question": "What is shown?",
                "ground_truth": "Logo",
                "image": "info.png",
                "sample_id": "k-1",
            }
        ],
        split="train",
    )
    assert len(samples) == 1
    assert samples[0].question == "What is shown?"
    assert samples[0].gold_answer == "Logo"


def test_infographicvqa_adapter_expands_due_document() -> None:
    adapter = build_dataset_adapter("infographicvqa")
    samples = adapter.convert_records(
        [
            {
                "name": "doc001",
                "annotations": [
                    {
                        "key": "Title?",
                        "metadata": {"question_id": "q1"},
                        "values": [{"value_variants": ["Annual Report", "report"]}],
                    },
                    {
                        "key": "Year?",
                        "metadata": {"question_id": "q2"},
                        "values": [{"value": "2024"}],
                    },
                ],
            }
        ],
        split="train",
        image_root="/data/iv",
    )
    assert len(samples) == 2
    assert samples[0].image_path == "/data/iv/doc001.png"
    assert samples[0].question == "Title?"
    assert samples[0].gold_answer == "Annual Report"
    assert samples[1].gold_answer == "2024"


def test_mmmu_adapter_uses_options_and_subset_metadata() -> None:
    adapter = build_dataset_adapter("mmmu")
    sample = adapter.convert_record(
        {
            "id": "validation_Accounting_1",
            "question": "Question: <image 1> Which option is correct?",
            "options": ["10", "20", "30"],
            "answer": "B",
            "image_1": "m1.png",
            "_hf_subset": "Accounting",
            "subfield": "Financial Accounting",
            "topic_difficulty": "Easy",
            "question_type": "multiple-choice",
        },
        index=0,
        split="validation",
    )
    assert sample.sample_id == "validation_Accounting_1"
    assert sample.choices == ["10", "20", "30"]
    assert sample.gold_answer == "B"
    assert sample.metadata["hf_subset"] == "Accounting"
    assert sample.metadata["topic_difficulty"] == "Easy"
