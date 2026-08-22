from pydantic import BaseModel, ConfigDict


class ItemBase(BaseModel):
    title: str
    completed: bool = False


class ItemCreate(ItemBase):
    pass


class ItemUpdate(BaseModel):
    title: str | None = None
    completed: bool | None = None


class ItemOut(ItemBase):
    id: int
    model_config = ConfigDict(from_attributes=True)
