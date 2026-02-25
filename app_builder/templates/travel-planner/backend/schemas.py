from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime


class TripBase(BaseModel):
    title: str
    destination: str
    start_date: datetime
    end_date: datetime
    budget: Optional[float] = None
    notes: Optional[str] = None


class TripCreate(TripBase):
    pass


class Trip(TripBase):
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)


class ItineraryItemBase(BaseModel):
    trip_id: int
    day: int
    title: str
    description: Optional[str] = None
    time_slot: Optional[str] = None


class ItineraryItemCreate(ItineraryItemBase):
    pass


class ItineraryItem(ItineraryItemBase):
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)
