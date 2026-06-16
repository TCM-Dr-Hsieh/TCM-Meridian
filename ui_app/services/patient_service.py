from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime

from ui_app.services.file_io import atomic_write_json, atomic_write_text


class PatientDataService:
    """File-backed patient and session storage."""

    def __init__(self, data_root: str):
        self.data_root = data_root

    def _build_folder_name(self, patient_id: str, birthday: str, name: str) -> str:
        return f"{patient_id}_{birthday}_{name}"

    @staticmethod
    def compact_summary(text: str, limit: int = 50) -> str:
        compact = " ".join((text or "").split())
        if len(compact) <= limit:
            return compact
        return compact[:limit]

    def _init_patient_info(
        self,
        patient_id: str,
        name: str,
        gender: str,
        birthday: str,
        phone: str,
        address: str,
        remark: str,
    ) -> dict:
        return {
            "basic_info": {
                "id": patient_id,
                "name": name,
                "gender": gender,
                "birthday": birthday,
                "phone": phone,
                "address": address,
                "remark": remark,
            },
            "directories": {
                "picture_row_dir": "Picture_Row/",
                "medical_info_dir": "Medical_information/",
                "log_dir": "log/",
            },
            "raw_images": [],
            "medical_information": {
                "Lab_Data": [],
                "Image_Data": [],
                "Special_Examination": [],
                "Medication_Profile": [],
                "Admission_Progress_Discharge_Note": [],
            },
            "sessions": {},
        }

    def create_patient(
        self,
        patient_id: str,
        name: str,
        gender: str,
        birthday: str,
        phone: str,
        address: str,
        remark: str,
    ) -> str:
        if not patient_id.strip():
            return "❌ 患者 ID 不可為空"
        if not name.strip():
            return "❌ 患者姓名不可為空"
        if not birthday.strip():
            return "❌ 生日不可為空"
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", birthday):
            return "❌ 生日格式錯誤，請使用 YYYY-MM-DD"

        folder_name = self._build_folder_name(patient_id.strip(), birthday.strip(), name.strip())
        folder_path = os.path.join(self.data_root, folder_name)

        if os.path.exists(folder_path):
            return f"❌ 患者資料夾已存在：{folder_name}"

        os.makedirs(folder_path, exist_ok=True)
        os.makedirs(os.path.join(folder_path, "Picture_Row"), exist_ok=True)
        os.makedirs(os.path.join(folder_path, "Medical_information"), exist_ok=True)
        os.makedirs(os.path.join(folder_path, "log"), exist_ok=True)

        info = self._init_patient_info(
            patient_id.strip(),
            name.strip(),
            gender.strip(),
            birthday.strip(),
            phone.strip(),
            address.strip(),
            remark.strip(),
        )
        json_path = os.path.join(folder_path, "patient_info.json")
        atomic_write_json(json_path, info)

        return folder_path

    def list_patients(self) -> list[dict]:
        results: list[dict] = []
        if not os.path.isdir(self.data_root):
            return results
        for entry in sorted(os.listdir(self.data_root)):
            entry_path = os.path.join(self.data_root, entry)
            json_path = os.path.join(entry_path, "patient_info.json")
            if os.path.isdir(entry_path) and os.path.isfile(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    results.append({
                        "folder_name": entry,
                        "folder_path": entry_path,
                        "basic_info": data.get("basic_info", {}),
                    })
                except Exception:
                    pass
        return results

    def load_patient(self, folder_path: str) -> dict | None:
        json_path = os.path.join(folder_path, "patient_info.json")
        if not os.path.isfile(json_path):
            return None
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_patient_info(self, folder_path: str, info: dict) -> str:
        json_path = os.path.join(folder_path, "patient_info.json")
        atomic_write_json(json_path, info)
        return "✅ 患者資料已儲存"

    def update_patient_basic_info(
        self,
        folder_path: str,
        new_id: str,
        new_name: str,
        new_gender: str,
        new_birthday: str,
        new_phone: str,
        new_address: str,
        new_remark: str,
    ) -> tuple[str, str]:
        info = self.load_patient(folder_path)
        if info is None:
            return "❌ 找不到患者資料", folder_path
        if not new_id.strip():
            return "❌ 患者 ID 不可為空", folder_path
        if not new_name.strip():
            return "❌ 患者姓名不可為空", folder_path
        if not new_birthday.strip():
            return "❌ 生日不可為空", folder_path
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", new_birthday):
            return "❌ 生日格式錯誤，請使用 YYYY-MM-DD", folder_path

        info["basic_info"] = {
            "id": new_id.strip(),
            "name": new_name.strip(),
            "gender": new_gender.strip(),
            "birthday": new_birthday.strip(),
            "phone": new_phone.strip(),
            "address": new_address.strip(),
            "remark": new_remark.strip(),
        }
        self.save_patient_info(folder_path, info)

        old_folder_name = os.path.basename(folder_path)
        new_folder_name = self._build_folder_name(new_id.strip(), new_birthday.strip(), new_name.strip())

        if old_folder_name != new_folder_name:
            new_folder_path = os.path.join(self.data_root, new_folder_name)
            if os.path.exists(new_folder_path):
                return f"❌ 無法重新命名：目標資料夾已存在 ({new_folder_name})", folder_path
            try:
                os.rename(folder_path, new_folder_path)
                return f"✅ 患者資料已更新，資料夾已重新命名為 {new_folder_name}", new_folder_path
            except Exception as e:
                return f"⚠️ 資料已儲存，但資料夾重新命名失敗：{e}", folder_path

        return "✅ 患者資料已更新", folder_path

    def delete_patient(self, folder_path: str) -> str:
        if not os.path.isdir(folder_path):
            return "❌ 找不到患者資料夾"
        try:
            folder_name = os.path.basename(folder_path)
            shutil.rmtree(folder_path)
            return f"✅ 已刪除患者：{folder_name}"
        except Exception as e:
            return f"❌ 刪除失敗：{e}"

    def list_sessions(self, folder_path: str) -> list[str]:
        info = self.load_patient(folder_path)
        if info is None:
            return []
        return sorted(info.get("sessions", {}).keys(), reverse=True)

    def create_session(self, folder_path: str, date_str: str, template_date: str = "") -> str:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return "❌ 日期格式錯誤，請使用 YYYY-MM-DD"
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return "❌ 無效日期，請檢查年月日是否合理"

        info = self.load_patient(folder_path)
        if info is None:
            return "❌ 找不到患者資料"

        if date_str in info.get("sessions", {}):
            return f"❌ 就診日期 {date_str} 已存在"

        note_file = f"{date_str}-NOTE.md"
        at_file = f"{date_str}-ASSESSMENT & TREATMENT.md"
        log_folder = f"log/{date_str}-log/"

        info.setdefault("sessions", {})[date_str] = {
            "note_file": note_file,
            "assessment_treatment_file": at_file,
            "log_folder": log_folder,
            "note_summary": "",
            "assessment_treatment_summary": "",
        }

        log_path = os.path.join(folder_path, "log", f"{date_str}-log")
        os.makedirs(log_path, exist_ok=True)

        note_path = os.path.join(folder_path, note_file)
        at_path = os.path.join(folder_path, at_file)

        if template_date and template_date in info.get("sessions", {}):
            tmpl_session = info["sessions"][template_date]
            tmpl_note = os.path.join(folder_path, tmpl_session.get("note_file", ""))
            tmpl_at = os.path.join(folder_path, tmpl_session.get("assessment_treatment_file", ""))
            if os.path.isfile(tmpl_note):
                shutil.copy2(tmpl_note, note_path)
            else:
                atomic_write_text(note_path, "")
            if os.path.isfile(tmpl_at):
                shutil.copy2(tmpl_at, at_path)
            else:
                atomic_write_text(at_path, "")
        else:
            atomic_write_text(note_path, "")
            atomic_write_text(at_path, "")

        self.save_patient_info(folder_path, info)
        return f"✅ 已建立就診日期 {date_str}"

    def delete_session(self, folder_path: str, date_str: str) -> str:
        info = self.load_patient(folder_path)
        if info is None:
            return "❌ 找不到患者資料"
        if date_str not in info.get("sessions", {}):
            return f"❌ 找不到就診日期 {date_str}"

        session = info["sessions"][date_str]

        for fkey in ["note_file", "assessment_treatment_file"]:
            fpath = os.path.join(folder_path, session.get(fkey, ""))
            if os.path.isfile(fpath):
                os.remove(fpath)

        log_dir = os.path.join(folder_path, "log", f"{date_str}-log")
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)

        del info["sessions"][date_str]
        self.save_patient_info(folder_path, info)
        return f"✅ 已刪除就診日期 {date_str}"

    def load_session_content(self, folder_path: str, date_str: str) -> dict:
        info = self.load_patient(folder_path)
        if info is None or date_str not in info.get("sessions", {}):
            return {"note": "", "at": ""}

        session = info["sessions"][date_str]
        note_path = os.path.join(folder_path, session.get("note_file", ""))
        at_path = os.path.join(folder_path, session.get("assessment_treatment_file", ""))

        note = ""
        at = ""
        if os.path.isfile(note_path):
            with open(note_path, "r", encoding="utf-8") as f:
                note = f.read()
        if os.path.isfile(at_path):
            with open(at_path, "r", encoding="utf-8") as f:
                at = f.read()

        return {"note": note, "at": at}

    def save_session_content(self, folder_path: str, date_str: str, note: str, at: str):
        info = self.load_patient(folder_path)
        if info is None or date_str not in info.get("sessions", {}):
            return
        session = info["sessions"][date_str]
        note_path = os.path.join(folder_path, session.get("note_file", ""))
        at_path = os.path.join(folder_path, session.get("assessment_treatment_file", ""))

        atomic_write_text(note_path, note)
        atomic_write_text(at_path, at)

    def get_session_summaries(self, folder_path: str, date_str: str) -> dict:
        info = self.load_patient(folder_path)
        if info is None or date_str not in info.get("sessions", {}):
            return {"note_summary": "", "assessment_treatment_summary": ""}

        session = info["sessions"][date_str]
        content = self.load_session_content(folder_path, date_str)
        return {
            "note_summary": session.get("note_summary") or self.compact_summary(content.get("note", "")),
            "assessment_treatment_summary": (
                session.get("assessment_treatment_summary")
                or self.compact_summary(content.get("at", ""))
            ),
        }

    def save_session_summaries(
        self,
        folder_path: str,
        date_str: str,
        note_summary: str,
        assessment_treatment_summary: str,
    ) -> str:
        info = self.load_patient(folder_path)
        if info is None or date_str not in info.get("sessions", {}):
            return "❌ 找不到就診日期"

        session = info["sessions"][date_str]
        session["note_summary"] = self.compact_summary(note_summary)
        session["assessment_treatment_summary"] = self.compact_summary(assessment_treatment_summary)
        self.save_patient_info(folder_path, info)
        return f"✅ 已儲存 {date_str} 病歷摘要"
