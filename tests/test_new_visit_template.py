"""A new visit started from an earlier visit's record gets that record as plain text: the source tags and the low-confidence marks
belong to the earlier visit's transcript, interview and evidence."""
from pathlib import Path

from ui_app.services.patient_service import PatientDataService

OLD_NOTE = ("甲- 現病史：頭痛三天[問診紀錄_患者R1]，**夜間加重（只有家屬轉述）**[問診紀錄_家屬R2]；發燒[語音#2-3]\n"
            "乙- 過去病史：高血壓十年[上次病歷] [人類醫師_手動修改]\n"
            "癸.2- 檢驗：HbA1c 7.3%[2026-10-01-Lab_data]；血壓 128/82 mmHg[人類醫師_T2]\n")
OLD_AT = ("A: 外感風熱[Forum_D3]\n"
          "T: 銀翹散 6g tid[人類醫師_T4]；**注意（孕婦慎用）**\n")


def make_patient(tmp_path):
    service = PatientDataService(str(tmp_path))
    folder = service.create_patient("A1", "王", "男", "1980-01-01", "", "", "")      # the new patient's folder
    assert Path(folder).is_dir()
    assert service.create_session(folder, "2026-10-01").startswith("✅")
    paths = service.load_patient(folder)["sessions"]["2026-10-01"]
    Path(folder, paths["note_file"]).write_text(OLD_NOTE, encoding="utf-8")
    Path(folder, paths["assessment_treatment_file"]).write_text(OLD_AT, encoding="utf-8")
    return service, folder


def test_a_new_visit_from_an_earlier_record_gets_its_words_without_its_tags_or_marks(tmp_path):
    service, folder = make_patient(tmp_path)
    assert service.create_session(folder, "2026-10-17", "2026-10-01").startswith("✅")
    content = service.load_session_content(folder, "2026-10-17")
    assert content["note"] == ("甲- 現病史：頭痛三天，夜間加重；發燒\n"
                               "乙- 過去病史：高血壓十年\n"
                               "癸.2- 檢驗：HbA1c 7.3%；血壓 128/82 mmHg\n")
    # the A&T loses its tags but keeps its own bold notes: they are not low-confidence marks
    assert content["at"] == "A: 外感風熱\nT: 銀翹散 6g tid；**注意（孕婦慎用）**\n"
    # the earlier visit itself is untouched
    assert service.load_session_content(folder, "2026-10-01") == {"note": OLD_NOTE, "at": OLD_AT}


def test_a_blank_start_and_a_missing_template_file_still_give_empty_records(tmp_path):
    service, folder = make_patient(tmp_path)
    assert service.create_session(folder, "2026-10-16").startswith("✅")                  # no template
    assert service.load_session_content(folder, "2026-10-16") == {"note": "", "at": ""}
    paths = service.load_patient(folder)["sessions"]["2026-10-01"]
    Path(folder, paths["assessment_treatment_file"]).unlink()                            # the template's A&T file is gone
    assert service.create_session(folder, "2026-10-17", "2026-10-01").startswith("✅")
    content = service.load_session_content(folder, "2026-10-17")
    assert content["at"] == "" and "頭痛三天" in content["note"] and "[" not in content["note"]
