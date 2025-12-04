# api_schema.py
from pydantic import BaseModel, Field

# Define the expected input payload structure with type hints and constraints
class IrisInput(BaseModel):
    sepal_length: float = Field(..., description="Sepal length in cm", ge=0)
    sepal_width: float = Field(..., description="Sepal width in cm", ge=0)
    petal_length: float = Field(..., description="Petal length in cm", ge=0)
    petal_width: float = Field(..., description="Petal width in cm", ge=0)

# Define the response structure
class PredictionOutput(BaseModel):
    class_name: str = Field(..., description="Predicted Iris species name")
    class_id: int = Field(..., description="Predicted class ID (0, 1, or 2)")
