"""Event store models (PRD §6.7, relational half).

An Event is one tracked object's visit: the same track ID on the same
camera, spanning first_seen..last_seen, with N individual Detections.

Identity rows are the relational half of the identity store: labels,
plates, appearance stats. The embeddings themselves live in the vector
store (Qdrant local mode, see siteloom/identity/vectors.py); the two are
joined by Identity.id, which is used as the vector point id's payload.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
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
    # True if the event falls inside a known guest arrival window
    # (booking correlation, PRD §6.7) — used to suppress false alarms.
    guest_window: Mapped[bool] = mapped_column(Boolean, default=False)

    camera: Mapped[Camera] = relationship(back_populates="events")
    detections: Mapped[list["Detection"]] = relationship(back_populates="event")
    identities: Mapped[list["EventIdentity"]] = relationship(back_populates="event")


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


class Identity(Base):
    """One recognized individual thing: a face, a person's appearance, a
    vehicle. `identifier_key` names which identification algorithm owns it
    ("face", "person", "vehicle", or a dynamically added class), so the
    same physical person can have both a face identity and an appearance
    identity — cross-linking them is a V1 concern.

    Label-and-learn (PRD §6.3): rows start with label=None and show up as
    "unknown-<id>" in the UI until the operator names them.
    """

    __tablename__ = "identities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    identifier_key: Mapped[str] = mapped_column(String, index=True)
    class_name: Mapped[str] = mapped_column(String, index=True)
    label: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    # Vehicle identities can also be matched by plate (PRD §6.4): plate OR
    # visual signature write to this same record, whichever is available.
    plate: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    first_seen: Mapped[datetime] = mapped_column(DateTime)
    last_seen: Mapped[datetime] = mapped_column(DateTime, index=True)
    appearance_count: Mapped[int] = mapped_column(Integer, default=0)
    # How many embeddings this identity has in the vector store (capped).
    vector_count: Mapped[int] = mapped_column(Integer, default=0)
    best_crop_path: Mapped[str | None] = mapped_column(String, nullable=True)

    events: Mapped[list["EventIdentity"]] = relationship(back_populates="identity")

    @property
    def display_name(self) -> str:
        return self.label or f"unknown-{self.identifier_key}-{self.id}"


class EventIdentity(Base):
    """Links an Event to the identities recognized during it."""

    __tablename__ = "event_identities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"), index=True)
    identity_id: Mapped[int] = mapped_column(ForeignKey("identities.id"), index=True)
    similarity: Mapped[float] = mapped_column(Float, default=0.0)
    hit_count: Mapped[int] = mapped_column(Integer, default=1)

    event: Mapped[Event] = relationship(back_populates="identities")
    identity: Mapped[Identity] = relationship(back_populates="events")


class NoiseEvent(Base):
    """A sustained loud episode (PRD §6.5, NoiseAware/Minut model).

    Levels are dBFS (relative to digital full scale), not SPL — thresholds
    are calibrated per microphone/deployment, not absolute loudness.
    """

    __tablename__ = "noise_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), index=True)
    start: Mapped[datetime] = mapped_column(DateTime, index=True)
    end: Mapped[datetime] = mapped_column(DateTime)
    peak_db: Mapped[float] = mapped_column(Float)
    mean_db: Mapped[float] = mapped_column(Float)


class Booking(Base):
    """A guest booking synced from iCal (PRD §6.7 guest-correlation)."""

    __tablename__ = "bookings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    uid: Mapped[str] = mapped_column(String, unique=True, index=True)
    summary: Mapped[str] = mapped_column(String, default="")
    start: Mapped[datetime] = mapped_column(DateTime, index=True)
    end: Mapped[datetime] = mapped_column(DateTime, index=True)
