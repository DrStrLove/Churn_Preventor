#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base

class User(Base):
    __tablename__ = "users"
    id    = Column(Integer, primary_key=True)
    name  = Column(String, unique=True, index=True)
    prefs = Column(JSON)   # e.g. preferred categories

    sessions = relationship("Session", back_populates="user")

class Session(Base):
    __tablename__ = "sessions"
    id          = Column(String, primary_key=True)
    user_id     = Column(Integer, ForeignKey("users.id"))
    started_at  = Column(DateTime)
    last_event  = Column(DateTime)
    churn_prob  = Column(Float)
    recommended = Column(String)  # JSON list or CSV string of actions
    completed   = Column(Boolean, default=False)

    user = relationship("User", back_populates="sessions")

class Event(Base):
    __tablename__ = "events"
    id         = Column(Integer, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"))
    ts         = Column(DateTime)
    raw_action = Column(String)
    action_id  = Column(Integer)
    p_flow     = Column(Float)

    session = relationship("Session", back_populates="events")

