from pydantic import BaseModel, Field
from typing import Optional
from datetime import date

class SalesRow(BaseModel):
    Date: date
    SKU_ID: str
    Sales_Channel: str
    Sales_Quantity: float = Field(ge=0)
    Price: Optional[float] = None
    Promotion_Active: Optional[int] = None
    Discount_Percentage: Optional[float] = None
    Holiday_Flag: Optional[int] = None

class InventoryRow(BaseModel):
    Date: date
    SKU_ID: str
    Current_Stock: float = Field(ge=0)

class LeadTimeRow(BaseModel):
    SKU_ID: str
    Lead_Time_Days: int = Field(gt=0)
    Order_Multiple: Optional[int] = 1
    MOQ: Optional[int] = None
    Shelf_Life_Days: Optional[int] = None

class BOMRow(BaseModel):
    Parent_SKU: str
    Component_SKU: str
    Qty_Per: float = Field(gt=0)

class GlobalConfig(BaseModel):
    Holding_Cost_Per_Unit_Per_Day: float = Field(gt=0)
    Ordering_Cost_Per_Order: float = Field(gt=0)
    Default_Service_Level: float = Field(gt=0, le=0.999)
