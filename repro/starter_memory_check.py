"""Run the starter's Liar's Bar series runner with a fake LLM, to check the --enable-memory path."""
import json, random, re, sys, types, runpy
random.seed(0)
class LLMClient:
    def __init__(self): pass
    def chat(self, messages, model="x", max_new_tokens=0):
        p = messages[-1]["content"]
        if "PARTNER:" in p or "ACCEPT" in p:
            names = [n for n in ["Lily", "Luke", "Mike", "Quinn"] if n in p]
            return f"ACCEPT - PARTNER: {names[-1]} - ok", ""
        if "played_cards" in p:
            hand = re.search(r"Your current hand is: (.+)", p).group(1).split(", ")
            return json.dumps({"played_cards": hand[:1], "behavior": "calm", "play_reason": "r"}), ""
        if "was_challenged" in p:
            return json.dumps({"was_challenged": random.random() < 0.4, "challenge_reason": "r"}), ""
        return json.dumps({"opinion": "ok"}), ""
mod = types.ModuleType("src.llm.llm_client"); mod.LLMClient = LLMClient
sys.modules["src.llm.llm_client"] = mod
sys.argv = ["x", "--games", "3", "--trigger-game", "1", "--trigger-round", "1", "--chosen-player", "Lily", "--enable-memory"]
runpy.run_path("src/game/game_series_secret_channel.py", run_name="__main__")
