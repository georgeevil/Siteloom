"""Event store models (PRD §6.7, relational half).

An Event is one tracked object's visit: the same track ID on the same
camera, spanning first_seen..last_seen, with N individual Detections.
The vector store for face/vehicle embeddings arrives with those modules;
this schema only carries what the vertical slice produces.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    site_id: Mapped[str] = mapped_column(String, index=True)
    name: Mapped[str] = mapped_column(String, default="")
    adapter: Mapped[str] = mapped_column(String, default="")

    events: Mapped[list["Event"]] = relationship(back_populates="camera")


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), index=True)
    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    class_name: Mapped[str] = mapped_column(String, index=True)
    first_seen: Mapped[datetime] = mapped_column(DateTime, index=True)
    last_seen: Mapped[datetime] = mapped_column(DateTime, index=True)
    detection_count: Mapped[int] = mapped_column(Integer, default=0)
    # Highest-confidence crop for this event — the thumbnail.
    best_crop_path: Mapped[str | None] = mapped_column(String, nullable=True)
    best_confidence: Mapped[float] = mapped_column(Float, default=0.0)

    camera: Mapped[Camera] = relationship(back_populates="events")
    detections: Mapped[list["Detection"]] = relationship(back_populates="event")


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    class_name: Mapped[str] = mapped_column(String)
    confidence: Mapped[float] = mapped_column(Float)
    bbox: Mapped[str] = mapped_column(Text)  # JSON [x1, y1, x2, y2]
    zones: Mapped[str] = mapped_column(Text, default="[]")  # JSON [name, ...]
    crop_path: Mapped[str | None] = mapped_column(String, nullable=True)

    event: Mapped[Event] = relationship(back_populates="detections")
