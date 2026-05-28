#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class EventIn(BaseModel):
    client_id: str
    watch_tms: datetime
    goal_nm_lvl1: str
    device_id: int

class RecommendOut(BaseModel):
    session_id: str
    churn_prob: float
    recommended_actions: List[str]

