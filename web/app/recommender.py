import joblib
import numpy as np
from pathlib import Path
from d3rlpy import load_learnable

# Папка, куда офлайн-скрипт сохраняет артефакты
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

# 1) HMM + энкодер
hmm_pack = joblib.load(ARTIFACTS / "hmm.pkl")
flow_id  = joblib.load(ARTIFACTS / "flow_id.pkl")

# 2) Загружаем полного CQL-агента из .d3 (config + веса)
cql_model = load_learnable(str(ARTIFACTS / "cql_model.d3"), device="cpu")

# 3) Список действий в том же порядке, в каком обучался агент
ACTION_LIST = [
    "CATALOG",
    "SEARCH",
    "CARD",
    "ARTICLES",
    "BREADCRUMBS",
    "MAIN",
    "CART_VIEW",
    "CABINET",
    "OTHER"
]

def map_action(raw: str) -> str:
    t = raw.lower()
    if "каталог" in t:          return "CATALOG"
    if "поиск"   in t:          return "SEARCH"
    if "карточ"  in t:          return "CARD"
    if "статей"  in t:          return "ARTICLES"
    if "breadcrumbs" in t or "хлебн" in t: return "BREADCRUMBS"
    if "главн"   in t:          return "MAIN"
    if "корзин"  in t:          return "CART_VIEW"
    if "кабинет" in t:          return "CABINET"
    return "OTHER"

def process_session(events):
    """
    Принимает упорядоченный список событий одной сессии (каждое имеет .goal_nm_lvl1).
    Возвращает:
      churn_prob: float  — вероятность Flow на последнем шаге
      recommended: str  — рекомендованное действие
    """
    # 1) raw_action → action_id
    raws = [e.goal_nm_lvl1 for e in events]
    acts = [map_action(r)      for r in raws]
    a_ids= np.array([ACTION_LIST.index(a) for a in acts], dtype=int).reshape(-1, 1)

    # 2) HMM → вектор proba_seq (T×n_states)
    hmm       = hmm_pack['hmm']
    proba_seq = hmm.predict_proba(a_ids)
    churn_prob= float(proba_seq[-1, flow_id])

    # 3) скрытое состояние последнего шага
    state = int(hmm.predict(a_ids)[-1])

    # 4) RL → argmax Q(s,a)
    q_vals      = cql_model.predict_value(np.array([[state]], dtype=np.float32))[0]
    recommended = ACTION_LIST[int(np.argmax(q_vals))]

    return churn_prob, recommended
