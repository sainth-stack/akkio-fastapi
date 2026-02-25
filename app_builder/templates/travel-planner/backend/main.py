from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import models
import schemas
import database

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/trips", response_model=list[schemas.Trip])
def list_trips(skip: int = 0, limit: int = 100, db: Session = Depends(database.get_db)):
    return db.query(models.Trip).offset(skip).limit(limit).all()


@app.post("/trips", response_model=schemas.Trip)
def create_trip(item: schemas.TripCreate, db: Session = Depends(database.get_db)):
    trip = models.Trip(**item.model_dump())
    db.add(trip)
    db.commit()
    db.refresh(trip)
    return trip


@app.get("/trips/{id}", response_model=schemas.Trip)
def get_trip(id: int, db: Session = Depends(database.get_db)):
    trip = db.query(models.Trip).filter(models.Trip.id == id).first()
    if not trip:
        raise HTTPException(status_code=404)
    return trip


@app.put("/trips/{id}", response_model=schemas.Trip)
def update_trip(id: int, item: schemas.TripCreate, db: Session = Depends(database.get_db)):
    trip = db.query(models.Trip).filter(models.Trip.id == id).first()
    if not trip:
        raise HTTPException(status_code=404)
    for k, v in item.model_dump().items():
        setattr(trip, k, v)
    db.commit()
    db.refresh(trip)
    return trip


@app.delete("/trips/{id}")
def delete_trip(id: int, db: Session = Depends(database.get_db)):
    trip = db.query(models.Trip).filter(models.Trip.id == id).first()
    if not trip:
        raise HTTPException(status_code=404)
    db.query(models.ItineraryItem).filter(models.ItineraryItem.trip_id == id).delete()
    db.delete(trip)
    db.commit()
    return {"ok": True}


@app.get("/itinerary", response_model=list[schemas.ItineraryItem])
def list_itinerary(skip: int = 0, limit: int = 200, trip_id: int = Query(None), db: Session = Depends(database.get_db)):
    q = db.query(models.ItineraryItem)
    if trip_id is not None:
        q = q.filter(models.ItineraryItem.trip_id == trip_id)
    return q.offset(skip).limit(limit).all()


@app.post("/itinerary", response_model=schemas.ItineraryItem)
def create_itinerary(item: schemas.ItineraryItemCreate, db: Session = Depends(database.get_db)):
    itinerary = models.ItineraryItem(**item.model_dump())
    db.add(itinerary)
    db.commit()
    db.refresh(itinerary)
    return itinerary


@app.get("/itinerary/{id}", response_model=schemas.ItineraryItem)
def get_itinerary(id: int, db: Session = Depends(database.get_db)):
    item = db.query(models.ItineraryItem).filter(models.ItineraryItem.id == id).first()
    if not item:
        raise HTTPException(status_code=404)
    return item


@app.put("/itinerary/{id}", response_model=schemas.ItineraryItem)
def update_itinerary(id: int, item: schemas.ItineraryItemCreate, db: Session = Depends(database.get_db)):
    itinerary = db.query(models.ItineraryItem).filter(models.ItineraryItem.id == id).first()
    if not itinerary:
        raise HTTPException(status_code=404)
    for k, v in item.model_dump().items():
        setattr(itinerary, k, v)
    db.commit()
    db.refresh(itinerary)
    return itinerary


@app.delete("/itinerary/{id}")
def delete_itinerary(id: int, db: Session = Depends(database.get_db)):
    item = db.query(models.ItineraryItem).filter(models.ItineraryItem.id == id).first()
    if not item:
        raise HTTPException(status_code=404)
    db.delete(item)
    db.commit()
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5001, reload=True)
