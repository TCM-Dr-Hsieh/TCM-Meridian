from __future__ import annotations


class AgentFactory:
    """Create agent instances from the existing config schema."""

    def create_main_agent(self, cfg: dict):
        from Main_Agent import MainAgent

        return MainAgent(
            cfg.get("main_agent", {}),
            cfg.get("record_subagent", {}),
            cfg.get("hallucination_subagent", None),
            cfg.get("ic_subagent", None),
            cfg.get("lc_subagent", None),
            cfg.get("nr_subagent", None),
            cfg.get("professor_config", None),
        )

    def create_main_agent_if_configured(self, cfg: dict):
        main_cfg = cfg.get("main_agent", {})
        if not main_cfg.get("model_name"):
            return None
        return self.create_main_agent(cfg)

    def create_information_collection_agent(self, cfg: dict):
        from Information_Collection_Subagent import InformationCollectionSubagent

        return InformationCollectionSubagent(cfg)

    def create_information_collection_agent_if_configured(self, cfg: dict):
        if not cfg.get("model_name"):
            return None
        return self.create_information_collection_agent(cfg)
