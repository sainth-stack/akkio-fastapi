from fastapi import FastAPI, Depends, HTTPException
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


@app.get("/todos", response_model=list[schemas.Todo])
def list_todos(skip: int = 0, limit: int = 100, db: Session = Depends(database.get_db)):
    return db.query(models.Todo).offset(skip).limit(limit).all()


@app.post("/todos", response_model=schemas.Todo)
def create_todo(item: schemas.TodoCreate, db: Session = Depends(database.get_db)):
    todo = models.Todo(**item.model_dump())
    db.add(todo)
    db.commit()
    db.refresh(todo)
    return todo


@app.get("/todos/{id}", response_model=schemas.Todo)
def get_todo(id: int, db: Session = Depends(database.get_db)):
    todo = db.query(models.Todo).filter(models.Todo.id == id).first()
    if not todo:
        raise HTTPException(status_code=404)
    return todo


@app.put("/todos/{id}", response_model=schemas.Todo)
def update_todo(id: int, item: schemas.TodoCreate, db: Session = Depends(database.get_db)):
    todo = db.query(models.Todo).filter(models.Todo.id == id).first()
    if not todo:
        raise HTTPException(status_code=404)
    for k, v in item.model_dump().items():
        setattr(todo, k, v)
    db.commit()
    db.refresh(todo)
    return todo


@app.delete("/todos/{id}")
def delete_todo(id: int, db: Session = Depends(database.get_db)):
    todo = db.query(models.Todo).filter(models.Todo.id == id).first()
    if not todo:
        raise HTTPException(status_code=404)
    db.delete(todo)
    db.commit()
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5001, reload=True)
