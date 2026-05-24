from dotenv import load_dotenv
load_dotenv()

import os
import io
import uuid
import json
import zipfile
import logging
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Literal, Dict, Any

import bcrypt
import jwt
import requests
import pandas as pd
from fastapi import FastAPI, APIRouter, HTTPException, Request, Response, Depends, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr, Field
from motor.motor_asyncio import AsyncIOMotorClient

try:
    from emergentintegrations.llm.chat import LlmChat, UserMessage
except ImportError:
    LlmChat = None
    UserMessage = None

# ===========================================================================
# Setup
# ===========================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("banhbao")

MONGO_URL = os.environ["MONGO_URL"]
DB_NAME = os.environ["DB_NAME"]
JWT_SECRET = os.environ["JWT_SECRET"]
JWT_ALG = "HS256"
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@banhbao.vn")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
EMERGENT_LLM_KEY = os.environ.get("EMERGENT_LLM_KEY", "")
SEED_DEMO = os.environ.get("SEED_DEMO", "false").lower() == "true"

mongo_client = AsyncIOMotorClient(MONGO_URL)
db = mongo_client[DB_NAME]

app = FastAPI(title="Bánh Bao Admin API v2")
api = APIRouter(prefix="/api")

cors_origins_env = os.environ.get("CORS_ORIGINS", "*")
if cors_origins_env.strip() == "*":
    cors_origins = ["*"]
    allow_credentials = False
else:
    cors_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
    allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================================================================
# Helpers
# ===========================================================================
ROLES = ("admin", "manager", "coordinator", "warehouse", "accountant", "shipper", "staff")

# Granular permission keys
PERMISSION_KEYS = (
    "orders.view", "orders.create", "orders.edit", "orders.delete", "orders.print",
    "products.view", "products.create", "products.edit", "products.delete",
    "materials.view", "materials.create", "materials.edit", "materials.delete",
    "stock.in", "stock.out", "stock.adjust",
    "customers.view", "customers.create", "customers.edit", "customers.delete", "customers.import",
    "suppliers.view", "suppliers.create", "suppliers.edit", "suppliers.delete",
    "debts.view", "debts.collect", "debts.pay", "debts.remind",
    "delivery.view", "delivery.assign", "delivery.bill",
    "reports.view", "reports.export",
    "users.view", "users.create", "users.edit", "users.delete",
    "settings.view", "settings.edit",
)

# Default permissions matrix per role
DEFAULT_PERMISSIONS: Dict[str, List[str]] = {
    "admin": list(PERMISSION_KEYS),  # full
    "manager": [k for k in PERMISSION_KEYS if not k.startswith("users.") and not k.startswith("settings.edit")] + ["settings.view"],
    "coordinator": [
        "orders.view", "orders.create", "orders.edit", "orders.print",
        "products.view", "customers.view", "customers.create", "customers.edit",
        "delivery.view", "delivery.assign", "delivery.bill",
        "debts.view", "reports.view",
    ],
    "warehouse": [
        "products.view", "products.create", "products.edit",
        "materials.view", "materials.create", "materials.edit",
        "stock.in", "stock.out", "stock.adjust",
        "suppliers.view", "suppliers.create", "suppliers.edit",
        "reports.view",
    ],
    "accountant": [
        "orders.view", "orders.print", "customers.view", "suppliers.view",
        "debts.view", "debts.collect", "debts.pay", "debts.remind",
        "reports.view", "reports.export",
    ],
    "shipper": [
        "orders.view", "delivery.view", "delivery.bill",
    ],
    "staff": [
        "orders.view", "orders.create", "products.view", "customers.view", "customers.create",
        "reports.view",
    ],
}


VN_TZ = timezone(timedelta(hours=7))

def now_vn():
    """Trả về giờ hiện tại theo múi giờ Việt Nam (UTC+7)."""
    return datetime.now(VN_TZ)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def _ensure_dt(value):
    """Coerce a value into a timezone-aware datetime. Handles str (ISO format) +
    naive datetime + None. Returns None if value can't be parsed."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None
    if hasattr(value, "tzinfo") and value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def create_access_token(user_id: str, email: str, role: str) -> str:
    payload = {
        "sub": user_id, "email": email, "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=12),
        "type": "access",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def create_refresh_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(days=7),
        "type": "refresh",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def set_auth_cookies(resp: Response, access: str, refresh: str):
    resp.set_cookie("access_token", access, httponly=True, secure=True, samesite="none", max_age=12*3600, path="/")
    resp.set_cookie("refresh_token", refresh, httponly=True, secure=True, samesite="none", max_age=7*24*3600, path="/")


def clear_auth_cookies(resp: Response):
    for c in ("access_token", "refresh_token", "session_token"):
        resp.delete_cookie(c, path="/")


async def get_token_from_request(request: Request) -> Optional[str]:
    tok = request.cookies.get("access_token")
    if tok:
        return tok
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


async def get_current_user(request: Request) -> dict:
    token = await get_token_from_request(request)
    if token:
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
            if payload.get("type") == "access":
                user = await db.users.find_one({"user_id": payload["sub"]}, {"_id": 0, "password_hash": 0})
                if user:
                    return user
        except jwt.ExpiredSignatureError:
            pass
        except jwt.InvalidTokenError:
            pass

    session_token = request.cookies.get("session_token") or request.headers.get("X-Session-Token")
    if session_token:
        session = await db.user_sessions.find_one({"session_token": session_token}, {"_id": 0})
        if session:
            expires_at = session.get("expires_at")
            if isinstance(expires_at, str):
                expires_at = datetime.fromisoformat(expires_at)
            if expires_at and expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if not expires_at or expires_at > datetime.now(timezone.utc):
                user = await db.users.find_one({"user_id": session["user_id"]}, {"_id": 0, "password_hash": 0})
                if user:
                    return user

    raise HTTPException(status_code=401, detail="Not authenticated")


def get_user_permissions(user: dict) -> List[str]:
    """Custom permissions stored on user override the role default."""
    custom = user.get("permissions")
    if isinstance(custom, list) and custom:
        return custom
    return DEFAULT_PERMISSIONS.get(user.get("role", "staff"), DEFAULT_PERMISSIONS["staff"])


def require_role(*roles: str):
    async def dep(user: dict = Depends(get_current_user)) -> dict:
        if user.get("role") not in roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return dep


def require_permission(*keys: str):
    """Require ALL given permission keys (or admin role)."""
    async def dep(user: dict = Depends(get_current_user)) -> dict:
        if user.get("role") == "admin":
            return user
        perms = set(get_user_permissions(user))
        for k in keys:
            if k not in perms:
                raise HTTPException(status_code=403, detail=f"Missing permission: {k}")
        return user
    return dep


# ===========================================================================
# PIN2 — second-level password required to mutate orders (delete/edit/status)
# ===========================================================================
async def _get_pin2_hash() -> Optional[str]:
    sec = await db.security_settings.find_one({"_singleton": "sec"}, {"_id": 0})
    return sec.get("pin2_hash") if sec else None


async def require_pin2(request: Request, user: dict = Depends(get_current_user)) -> dict:
    """If a pin2 is set in security_settings, demand a matching X-PIN2 header."""
    pin2_hash = await _get_pin2_hash()
    if not pin2_hash:
        return user
    provided = request.headers.get("X-PIN2") or ""
    if not provided or not verify_password(provided, pin2_hash):
        raise HTTPException(status_code=403, detail="PIN2_REQUIRED")
    return user


def require_permission_and_pin2(*keys: str):
    """Combined dependency: permission + (if set) PIN2 header validation."""
    base = require_permission(*keys)

    async def dep(request: Request, user: dict = Depends(base)) -> dict:
        pin2_hash = await _get_pin2_hash()
        if pin2_hash:
            provided = request.headers.get("X-PIN2") or ""
            if not provided or not verify_password(provided, pin2_hash):
                raise HTTPException(status_code=403, detail="PIN2_REQUIRED")
        return user

    return dep


def _normalize_payment(method: Optional[str]) -> str:
    """Backwards-compat: 'cod' is renamed to 'debt'."""
    if method == "cod":
        return "debt"
    return method or "cash"


# ===========================================================================
# Models
# ===========================================================================
class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    name: str = Field(min_length=1, max_length=80)
    role: Literal["admin", "manager", "coordinator", "warehouse", "accountant", "shipper", "staff"] = "staff"
    permissions: Optional[List[str]] = None
    phone: Optional[str] = ""


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class UserUpdateIn(BaseModel):
    name: Optional[str] = None
    role: Optional[Literal["admin", "manager", "coordinator", "warehouse", "accountant", "shipper", "staff"]] = None
    permissions: Optional[List[str]] = None
    phone: Optional[str] = None
    payment_method: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None


class ProductIn(BaseModel):
    name: str
    sku: Optional[str] = None
    category: str = "Bánh bao"
    description: Optional[str] = ""
    price: float
    wholesale_price: float = 0
    cost: float = 0
    stock: int = 0
    low_stock_threshold: int = 10
    unit: str = "cái"
    image_url: Optional[str] = None
    variants: List[dict] = []
    is_active: bool = True
    expiration_days: int = 3  # default shelf life for products


class MaterialIn(BaseModel):
    name: str
    code: Optional[str] = None
    unit: str = "kg"
    stock: float = 0
    cost: float = 0
    low_stock_threshold: float = 0
    expiration_days: int = 30  # default shelf life in days
    notes: Optional[str] = ""
    is_active: bool = True


class StockInIn(BaseModel):
    kind: Literal["material", "product"] = "material"
    target_id: str  # material_id or product_id
    quantity: float
    unit_price: float = 0
    supplier_id: Optional[str] = None
    supplier_name: Optional[str] = ""
    production_date: Optional[str] = None  # YYYY-MM-DD
    expiration_date: Optional[str] = None  # YYYY-MM-DD
    batch_code: Optional[str] = None
    notes: Optional[str] = ""


class StockAdjustIn(BaseModel):
    kind: Literal["material", "product"] = "product"
    target_id: str
    delta: float  # +/- adjustment
    reason: Literal["damage", "waste", "lost", "correction", "expired", "other"] = "correction"
    notes: Optional[str] = ""


class CustomerIn(BaseModel):
    name: str
    code: Optional[str] = ""
    nickname: Optional[str] = ""
    phone: Optional[str] = ""
    email: Optional[str] = ""
    address: Optional[str] = ""
    district: Optional[str] = ""
    city: Optional[str] = ""
    tax_id: Optional[str] = ""  # MST or CCCD
    group: str = "new"  # vip/regular/new + custom tags
    type: Literal["retail", "wholesale"] = "retail"
    classification: Optional[str] = ""  # free-text custom tag
    assigned_user_id: Optional[str] = ""  # nhân viên phụ trách giao hàng
    max_debt_days: int = 0
    max_debt_amount: float = 0
    notes: Optional[str] = ""


class SupplierIn(BaseModel):
    name: str
    code: Optional[str] = ""
    phone: Optional[str] = ""
    email: Optional[str] = ""
    address: Optional[str] = ""
    district: Optional[str] = ""
    city: Optional[str] = ""
    tax_id: Optional[str] = ""
    group: str = "default"
    rating: int = 5
    notes: Optional[str] = ""


class OrderItemIn(BaseModel):
    product_id: str
    name: str
    price: float
    quantity: int
    subtotal: float


class OrderIn(BaseModel):
    customer_id: Optional[str] = None
    customer_name: str
    customer_phone: Optional[str] = ""
    customer_address: Optional[str] = ""
    customer_district: Optional[str] = ""
    customer_city: Optional[str] = ""
    type: Literal["retail", "wholesale", "delivery"] = "retail"
    items: List[OrderItemIn]
    payment_method: str = "cash"  # cash/transfer/debt/ewallet/card
    note: Optional[str] = ""
    discount: float = 0
    discount_percent: float = 0
    shipping_fee: float = 0
    assigned_shipper_id: Optional[str] = ""
    due_date: Optional[str] = ""  # YYYY-MM-DD for debt orders
    remaining_amount: Optional[float] = 0  # số tiền còn nợ khi thanh toán 1 phần


class OrderStatusUpdate(BaseModel):
    status: Literal["new", "processing", "delivering", "delivered", "debt_pending", "cancelled"]
    note: Optional[str] = ""


class DebtPaymentIn(BaseModel):
    order_id: Optional[str] = None  # collect from customer order
    supplier_id: Optional[str] = None  # pay supplier
    customer_id: Optional[str] = None
    amount: float
    method: Literal["cash", "transfer", "ewallet", "card"] = "cash"
    notes: Optional[str] = ""


class ShopSettingsIn(BaseModel):
    shop_name: str = "Tiệm Bánh Bao"
    address: Optional[str] = ""
    phone: Optional[str] = ""
    email: Optional[str] = ""
    website: Optional[str] = ""
    tax_id: Optional[str] = ""
    bank_name: Optional[str] = ""
    bank_account: Optional[str] = ""
    bank_account_holder: Optional[str] = ""
    logo_url: Optional[str] = ""  # base64 or URL
    bill_show_logo: bool = True
    bill_show_address: bool = True
    bill_show_phone: bool = True
    bill_show_email: bool = False
    bill_show_website: bool = False
    bill_show_tax_id: bool = False
    bill_show_bank_qr: bool = True
    bill_footer_text: str = "Cảm ơn quý khách. Hẹn gặp lại!"
    default_language: Literal["vi", "en"] = "vi"
    # Bill customization (Đợt 3A)
    bill_accent_color: str = "#2D4A22"
    bill_logo_position: Literal["left", "center", "right"] = "left"
    bill_signature_customer: bool = False
    bill_signature_shipper: bool = False
    bill_signature_warehouse: bool = False
    bill_signature_accountant: bool = False
    bill_fixed_notes: List[str] = []  # up to 5 strings


class ChatIn(BaseModel):
    session_id: str
    message: str
    model: Optional[str] = None  # "claude" | "openai" | provider:model


# ===========================================================================
# System
# ===========================================================================
@api.get("/system/time")
async def system_time():
    now = now_vn()
    return {
        "utc": now.isoformat(),
        "vn": (now + timedelta(hours=7)).isoformat(),
        "timestamp_ms": int(now.timestamp() * 1000),
        "timezone": "UTC (host)",
    }


@api.get("/system/permissions")
async def system_permissions(user: dict = Depends(get_current_user)):
    return {
        "roles": list(ROLES),
        "keys": list(PERMISSION_KEYS),
        "defaults": DEFAULT_PERMISSIONS,
        "mine": get_user_permissions(user),
    }


@api.post("/system/reset-demo")
async def reset_demo(confirm: str = Query(...), user: dict = Depends(require_role("admin"))):
    """Wipe all transactional data (orders, customers, suppliers, products, materials, debts, chat).
    Pass ?confirm=YES_DELETE to confirm. Keeps users and settings.
    """
    if confirm != "YES_DELETE":
        raise HTTPException(status_code=400, detail="Pass confirm=YES_DELETE to proceed")
    collections = [
        "orders", "customers", "suppliers", "products", "materials",
        "stock_movements", "debt_payments", "chat_messages",
    ]
    counts = {}
    for c in collections:
        res = await db[c].delete_many({})
        counts[c] = res.deleted_count
    return {"ok": True, "deleted": counts}


# Đợt 3B — Export-all / Backup / Import-back
VN_HEADERS = {
    "customers": {
        "customer_id": "Mã KH", "code": "Mã hiển thị", "name": "Tên KH",
        "nickname": "Tên gọi", "phone": "SĐT", "email": "Email",
        "address": "Địa chỉ", "district": "Phường/Xã", "city": "Tỉnh/Thành",
        "tax_id": "MST/CCCD", "group": "Nhóm", "type": "Loại",
        "classification": "Phân loại", "max_debt_days": "Hạn nợ (ngày)",
        "max_debt_amount": "Hạn nợ (VND)", "notes": "Ghi chú",
        "total_orders": "Tổng đơn", "total_spent": "Tổng chi",
        "last_order_at": "Lần đặt cuối", "created_at": "Ngày tạo",
    },
    "products": {
        "product_id": "Mã SP", "sku": "SKU", "name": "Tên SP", "category": "Loại",
        "description": "Mô tả", "price": "Giá bán", "wholesale_price": "Giá sỉ",
        "cost": "Giá vốn", "stock": "Tồn kho", "unit": "ĐVT",
        "low_stock_threshold": "Ngưỡng tồn thấp", "expiration_days": "HSD mặc định (ngày)",
        "is_active": "Đang bán", "created_at": "Ngày tạo",
    },
    "orders": {
        "order_id": "Mã đơn", "order_code": "Mã hiển thị", "customer_id": "Mã KH",
        "customer_name": "Tên KH", "customer_phone": "SĐT KH",
        "customer_address": "Địa chỉ", "customer_district": "Phường/Xã",
        "customer_city": "Tỉnh/Thành", "type": "Loại đơn",
        "subtotal": "Tạm tính", "discount": "Giảm VND", "discount_percent": "Giảm %",
        "discount_amount": "Tổng giảm", "shipping_fee": "Phí ship",
        "total": "Tổng", "paid_amount": "Đã trả", "remaining_amount": "Còn nợ",
        "payment_method": "Thanh toán", "is_paid": "Đã trả?", "status": "Trạng thái",
        "due_date": "Hạn nợ", "assigned_shipper_id": "Mã shipper",
        "assigned_shipper_name": "Tên shipper", "note": "Ghi chú",
        "created_at": "Ngày đặt", "created_by": "Tạo bởi",
    },
    "debt_payments": {
        "payment_id": "Mã GD", "direction": "Hướng", "customer_id": "Mã KH",
        "customer_name": "Tên KH", "supplier_id": "Mã NCC", "supplier_name": "Tên NCC",
        "order_id": "Mã đơn", "amount": "Số tiền", "method": "Phương thức",
        "notes": "Ghi chú", "created_at": "Ngày GD", "created_by": "Tạo bởi",
    },
}


def _df_from_collection_items(items: List[Dict[str, Any]], headers_map: Dict[str, str]) -> pd.DataFrame:
    """Project ordered columns + rename to Vietnamese headers."""
    if not items:
        return pd.DataFrame(columns=list(headers_map.values()))
    rows = []
    for it in items:
        row = {}
        for key, vn_label in headers_map.items():
            v = it.get(key, "")
            # Format datetimes nicely
            if hasattr(v, "isoformat"):
                v = v.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(v, list):
                v = json.dumps(v, ensure_ascii=False, default=str)
            row[vn_label] = v
        rows.append(row)
    return pd.DataFrame(rows, columns=list(headers_map.values()))


@api.get("/system/export-all-excel")
async def export_all_excel(user: dict = Depends(require_role("admin"))):
    """Export all key collections into a single multi-sheet xlsx (Vietnamese headers)."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for coll, headers in VN_HEADERS.items():
            items = await db[coll].find({}, {"_id": 0}).limit(50000).to_list(None)
            df = _df_from_collection_items(items, headers)
            df.to_excel(writer, sheet_name=coll[:31], index=False)
    buf.seek(0)
    ts = now_vn().strftime("%Y%m%d_%H%M%S")
    filename = f"banhbao_export_{ts}.xlsx"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@api.get("/system/backup-zip")
async def backup_zip(user: dict = Depends(require_role("admin"))):
    """Stream a .zip containing JSON dumps of all collections (full backup)."""
    collections = [
        "users", "customers", "suppliers", "products", "materials",
        "orders", "stock_movements", "debt_payments", "settings",
        "production_batches", "security_settings",
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for coll in collections:
            items = await db[coll].find({}, {"_id": 0}).to_list(None)
            data = json.dumps(items, ensure_ascii=False, default=str, indent=2)
            zf.writestr(f"{coll}.json", data)
        meta = {
            "exported_at": now_vn().isoformat(),
            "by_user": user.get("email"),
            "collections": collections,
            "version": "v2.2",
        }
        zf.writestr("_meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
    buf.seek(0)
    ts = now_vn().strftime("%Y%m%d_%H%M%S")
    filename = f"banhbao_backup_{ts}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@api.post("/system/import-all-excel")
async def import_all_excel(file: UploadFile = File(...), user: dict = Depends(require_role("admin"))):
    """Read a multi-sheet xlsx exported earlier and upsert rows into each known collection.

    The xlsx must use the Vietnamese headers produced by /system/export-all-excel.
    Rows are matched by the entity *_id field (customer_id, product_id, order_id, payment_id).
    Missing IDs are auto-generated.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="File rỗng")
    counts: Dict[str, int] = {}
    try:
        for coll, headers_map in VN_HEADERS.items():
            try:
                df = pd.read_excel(io.BytesIO(contents), sheet_name=coll[:31])
            except Exception:
                continue
            inv = {v: k for k, v in headers_map.items()}  # VN label -> internal key
            id_field = {
                "customers": "customer_id", "products": "product_id",
                "orders": "order_id", "debt_payments": "payment_id",
            }[coll]
            inserted = 0
            for _, row in df.iterrows():
                doc: Dict[str, Any] = {}
                for vn_label in df.columns:
                    key = inv.get(vn_label) or vn_label
                    v = row[vn_label]
                    if pd.isna(v):
                        continue
                    doc[key] = v
                if not doc.get(id_field):
                    doc[id_field] = f"{id_field.split('_')[0][:3]}_{uuid.uuid4().hex[:10]}"
                # Upsert
                await db[coll].update_one(
                    {id_field: doc[id_field]},
                    {"$set": doc},
                    upsert=True,
                )
                inserted += 1
            counts[coll] = inserted
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi đọc file: {e}")
    return {"ok": True, "imported": counts}


# ===========================================================================
# Auth
# ===========================================================================
@api.post("/auth/register")
async def register(body: RegisterIn, response: Response):
    email = body.email.lower().strip()
    if await db.users.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="Email đã tồn tại")
    user_id = f"user_{uuid.uuid4().hex[:12]}"
    doc = {
        "user_id": user_id, "email": email, "name": body.name, "role": body.role,
        "phone": body.phone or "",
        "password_hash": hash_password(body.password),
        "auth_provider": "local",
        "is_active": True,
        "permissions": body.permissions if body.permissions is not None else None,
        "created_at": now_vn(),
    }
    await db.users.insert_one(doc)
    access = create_access_token(user_id, email, body.role)
    refresh = create_refresh_token(user_id)
    set_auth_cookies(response, access, refresh)
    doc.pop("password_hash", None)
    doc.pop("_id", None)
    return doc


@api.post("/auth/login")
async def login(body: LoginIn, request: Request, response: Response):
    email = body.email.lower().strip()
    ip = request.client.host if request.client else "unknown"
    identifier = f"{ip}:{email}"

    attempt = await db.login_attempts.find_one({"identifier": identifier})
    if attempt and attempt.get("count", 0) >= 5:
        locked_until = attempt.get("locked_until")
        if locked_until and locked_until.tzinfo is None:
            locked_until = locked_until.replace(tzinfo=timezone.utc)
        if locked_until and locked_until > datetime.now(timezone.utc):
            raise HTTPException(status_code=429, detail="Quá nhiều lần thử. Thử lại sau 15 phút.")

    user = await db.users.find_one({"email": email})
    if not user or not user.get("password_hash") or not verify_password(body.password, user["password_hash"]):
        await db.login_attempts.update_one(
            {"identifier": identifier},
            {"$inc": {"count": 1}, "$set": {"locked_until": datetime.now(timezone.utc) + timedelta(minutes=15)}},
            upsert=True,
        )
        raise HTTPException(status_code=401, detail="Email hoặc mật khẩu không đúng")

    if user.get("is_active") is False:
        raise HTTPException(status_code=403, detail="Tài khoản đã bị khóa")

    await db.login_attempts.delete_one({"identifier": identifier})
    access = create_access_token(user["user_id"], email, user.get("role", "staff"))
    refresh = create_refresh_token(user["user_id"])
    set_auth_cookies(response, access, refresh)
    user.pop("password_hash", None)
    user.pop("_id", None)
    return user


@api.post("/auth/logout")
async def logout(request: Request, response: Response):
    session_token = request.cookies.get("session_token")
    if session_token:
        await db.user_sessions.delete_one({"session_token": session_token})
    clear_auth_cookies(response)
    return {"ok": True}


@api.get("/auth/me")
async def me(user: dict = Depends(get_current_user)):
    user["permissions_effective"] = get_user_permissions(user)
    return user


@api.post("/auth/refresh")
async def refresh_token(request: Request, response: Response):
    rt = request.cookies.get("refresh_token")
    if not rt:
        raise HTTPException(status_code=401, detail="No refresh token")
    try:
        payload = jwt.decode(rt, JWT_SECRET, algorithms=[JWT_ALG])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user = await db.users.find_one({"user_id": payload["sub"]}, {"_id": 0, "password_hash": 0})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        new_access = create_access_token(user["user_id"], user["email"], user.get("role", "staff"))
        response.set_cookie("access_token", new_access, httponly=True, secure=True, samesite="none", max_age=12*3600, path="/")
        return {"ok": True}
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")


@api.post("/auth/session")
async def emergent_session(request: Request, response: Response):
    session_id = request.headers.get("X-Session-ID") or (await request.json()).get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")
    try:
        r = requests.get(
            "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
            headers={"X-Session-ID": session_id}, timeout=10,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Auth service error: {e}")
    if r.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid session_id")
    data = r.json()
    email = (data.get("email") or "").lower()
    name = data.get("name") or email.split("@")[0]
    picture = data.get("picture")
    session_token = data["session_token"]

    user = await db.users.find_one({"email": email})
    if not user:
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        user = {
            "user_id": user_id, "email": email, "name": name, "picture": picture,
            "role": "staff", "auth_provider": "google", "is_active": True,
            "created_at": now_vn(),
        }
        await db.users.insert_one(user)
    else:
        await db.users.update_one(
            {"email": email},
            {"$set": {"name": name, "picture": picture, "auth_provider": user.get("auth_provider", "google")}},
        )

    expires_at = now_vn() + timedelta(days=7)
    await db.user_sessions.update_one(
        {"session_token": session_token},
        {"$set": {
            "session_token": session_token, "user_id": user["user_id"],
            "expires_at": expires_at, "created_at": now_vn(),
        }},
        upsert=True,
    )
    response.set_cookie("session_token", session_token, httponly=True, secure=True, samesite="none", max_age=7*24*3600, path="/")
    user.pop("_id", None)
    user.pop("password_hash", None)
    return user


# ===========================================================================
# Settings
# ===========================================================================
SETTINGS_ID = "shop_settings_singleton"


@api.get("/settings")
async def get_settings(user: dict = Depends(get_current_user)):
    s = await db.settings.find_one({"_singleton": SETTINGS_ID}, {"_id": 0, "_singleton": 0})
    if not s:
        # return defaults
        defaults = ShopSettingsIn().model_dump()
        return defaults
    return s


@api.put("/settings")
async def update_settings(body: ShopSettingsIn, user: dict = Depends(require_permission("settings.edit"))):
    doc = body.model_dump()
    doc["updated_at"] = now_vn()
    await db.settings.update_one({"_singleton": SETTINGS_ID}, {"$set": doc, "$setOnInsert": {"_singleton": SETTINGS_ID}}, upsert=True)
    out = dict(doc)
    return out


# ===========================================================================
# Security — PIN2 + delete-all PIN pair (admin only)
# ===========================================================================
class Pin2SetIn(BaseModel):
    account_password: str
    new_pin: str = Field(min_length=4, max_length=32)


class DeletePinsSetIn(BaseModel):
    account_password: str
    pin_a: str = Field(min_length=4, max_length=32)
    pin_b: str = Field(min_length=4, max_length=32)


class Pin2VerifyIn(BaseModel):
    pin: str


class DeleteAllOrdersIn(BaseModel):
    account_password: str
    pin_a: str
    pin_b: str


async def _require_account_password(user: dict, password: str) -> dict:
    """Re-fetch user with password_hash and verify."""
    full = await db.users.find_one({"user_id": user["user_id"]})
    if not full or not full.get("password_hash") or not verify_password(password, full["password_hash"]):
        raise HTTPException(status_code=403, detail="Mật khẩu tài khoản không đúng")
    return full


@api.get("/security/status")
async def security_status(user: dict = Depends(get_current_user)):
    sec = await db.security_settings.find_one({"_singleton": "sec"}, {"_id": 0}) or {}
    return {
        "has_pin2": bool(sec.get("pin2_hash")),
        "has_delete_pins": bool(sec.get("delete_pin_a_hash")) and bool(sec.get("delete_pin_b_hash")),
        "pin2_updated_at": sec.get("pin2_updated_at"),
        "delete_pins_updated_at": sec.get("delete_pins_updated_at"),
    }


@api.post("/security/pin2")
async def set_pin2(body: Pin2SetIn, user: dict = Depends(require_role("admin"))):
    await _require_account_password(user, body.account_password)
    now = now_vn()
    await db.security_settings.update_one(
        {"_singleton": "sec"},
        {"$set": {"pin2_hash": hash_password(body.new_pin), "pin2_updated_at": now},
         "$setOnInsert": {"_singleton": "sec"}},
        upsert=True,
    )
    return {"ok": True}


@api.delete("/security/pin2")
async def clear_pin2(account_password: str = Query(...), user: dict = Depends(require_role("admin"))):
    await _require_account_password(user, account_password)
    await db.security_settings.update_one(
        {"_singleton": "sec"},
        {"$unset": {"pin2_hash": "", "pin2_updated_at": ""}},
    )
    return {"ok": True}


@api.post("/security/verify-pin2")
async def verify_pin2(body: Pin2VerifyIn, user: dict = Depends(get_current_user)):
    pin2_hash = await _get_pin2_hash()
    if not pin2_hash:
        return {"ok": True, "set": False}
    if not verify_password(body.pin, pin2_hash):
        raise HTTPException(status_code=403, detail="PIN2 không đúng")
    return {"ok": True, "set": True}


@api.post("/security/delete-pins")
async def set_delete_pins(body: DeletePinsSetIn, user: dict = Depends(require_role("admin"))):
    await _require_account_password(user, body.account_password)
    if body.pin_a == body.pin_b:
        raise HTTPException(status_code=400, detail="2 mật khẩu xóa phải khác nhau")
    now = now_vn()
    await db.security_settings.update_one(
        {"_singleton": "sec"},
        {"$set": {
            "delete_pin_a_hash": hash_password(body.pin_a),
            "delete_pin_b_hash": hash_password(body.pin_b),
            "delete_pins_updated_at": now,
        }, "$setOnInsert": {"_singleton": "sec"}},
        upsert=True,
    )
    return {"ok": True}


@api.post("/orders/delete-all")
async def delete_all_orders(body: DeleteAllOrdersIn, user: dict = Depends(require_role("admin"))):
    """3-layer protection: account password + 2 admin-set PINs."""
    sec = await db.security_settings.find_one({"_singleton": "sec"}, {"_id": 0}) or {}
    if not sec.get("delete_pin_a_hash") or not sec.get("delete_pin_b_hash"):
        raise HTTPException(status_code=400, detail="Chưa cài đặt 2 mật khẩu xóa. Vào Cài đặt → Bảo mật.")
    await _require_account_password(user, body.account_password)
    if not verify_password(body.pin_a, sec["delete_pin_a_hash"]):
        raise HTTPException(status_code=403, detail="Mật khẩu xóa A không đúng")
    if not verify_password(body.pin_b, sec["delete_pin_b_hash"]):
        raise HTTPException(status_code=403, detail="Mật khẩu xóa B không đúng")

    # Restore stock for non-cancelled orders before wiping
    orders = await db.orders.find({"status": {"$ne": "cancelled"}}, {"_id": 0, "items": 1}).to_list(None)
    for o in orders:
        for it in (o.get("items") or []):
            await db.products.update_one(
                {"product_id": it["product_id"]},
                {"$inc": {"stock": it.get("quantity", 0)}},
            )
    res = await db.orders.delete_many({})
    return {"ok": True, "deleted": res.deleted_count}


# ===========================================================================
# Dashboard
# ===========================================================================
@api.get("/dashboard/stats")
async def dashboard_stats(user: dict = Depends(get_current_user)):
    now = now_vn()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    today_orders = await db.orders.find(
        {"created_at": {"$gte": today_start}}, {"_id": 0, "total": 1, "status": 1}
    ).limit(2000).to_list(None)
    today_revenue = sum(o.get("total", 0) for o in today_orders if o.get("status") != "cancelled")

    total_products = await db.products.count_documents({"is_active": True})
    total_customers = await db.customers.count_documents({})

    debt_orders = await db.orders.find(
        {"payment_method": "debt", "status": {"$ne": "cancelled"}, "is_paid": {"$ne": True}},
        {"_id": 0, "total": 1},
    ).limit(1000).to_list(None)
    total_debt = sum(o.get("total", 0) for o in debt_orders)

    chart = []
    for i in range(6, -1, -1):
        day = today_start - timedelta(days=i)
        day_end = day + timedelta(days=1)
        day_orders = await db.orders.find(
            {"created_at": {"$gte": day, "$lt": day_end}, "status": {"$ne": "cancelled"}},
            {"_id": 0, "total": 1},
        ).limit(1000).to_list(None)
        chart.append({
            "date": day.strftime("%d/%m"),
            "revenue": sum(o.get("total", 0) for o in day_orders),
            "orders": len(day_orders),
        })

    low_stock = await db.products.find(
        {"$expr": {"$lte": ["$stock", "$low_stock_threshold"]}, "is_active": True}, {"_id": 0},
    ).sort("stock", 1).limit(8).to_list(None)

    recent_orders = await db.orders.find({}, {"_id": 0}).sort("created_at", -1).limit(6).to_list(None)

    pipeline = [
        {"$match": {"status": {"$ne": "cancelled"}}},
        {"$unwind": "$items"},
        {"$group": {
            "_id": "$items.product_id",
            "name": {"$first": "$items.name"},
            "quantity": {"$sum": "$items.quantity"},
            "revenue": {"$sum": "$items.subtotal"},
        }},
        {"$sort": {"quantity": -1}},
        {"$limit": 5},
    ]
    top_products = []
    async for r in db.orders.aggregate(pipeline):
        top_products.append({
            "product_id": r["_id"], "name": r["name"],
            "quantity": r["quantity"], "revenue": r["revenue"],
        })

    # Customers who haven't ordered for 14+ days
    cutoff = today_start - timedelta(days=14)
    inactive_pipeline = [
        {"$group": {"_id": "$customer_id", "last_order": {"$max": "$created_at"}}},
        {"$match": {"last_order": {"$lt": cutoff}}},
        {"$count": "n"},
    ]
    inactive_count = 0
    async for r in db.orders.aggregate(inactive_pipeline):
        inactive_count = r["n"]

    # Materials low stock count
    mat_low = await db.materials.count_documents({"$expr": {"$lte": ["$stock", "$low_stock_threshold"]}, "is_active": True})

    return {
        "today_revenue": today_revenue,
        "today_orders": len(today_orders),
        "total_debt": total_debt,
        "total_products": total_products,
        "total_customers": total_customers,
        "inactive_customers_14d": inactive_count,
        "low_stock_materials": mat_low,
        "chart_7days": chart,
        "low_stock_products": low_stock,
        "recent_orders": recent_orders,
        "top_products": top_products,
    }


# ===========================================================================
# Products
# ===========================================================================
@api.get("/products")
async def list_products(
    q: Optional[str] = None,
    category: Optional[str] = None,
    low_stock: bool = False,
    sort: str = "created_at",
    direction: Literal["asc", "desc"] = "desc",
    user: dict = Depends(get_current_user),
):
    query = {}
    if q:
        query["$or"] = [{"name": {"$regex": q, "$options": "i"}}, {"sku": {"$regex": q, "$options": "i"}}]
    if category:
        query["category"] = category
    if low_stock:
        query["$expr"] = {"$lte": ["$stock", "$low_stock_threshold"]}
    dirv = 1 if direction == "asc" else -1
    items = await db.products.find(query, {"_id": 0}).sort(sort, dirv).limit(500).to_list(None)
    return items


@api.post("/products")
async def create_product(body: ProductIn, user: dict = Depends(require_permission("products.create"))):
    doc = body.model_dump()
    doc["product_id"] = f"prd_{uuid.uuid4().hex[:10]}"
    doc["sku"] = doc.get("sku") or doc["product_id"].upper()
    doc["created_at"] = now_vn()
    doc["updated_at"] = doc["created_at"]
    await db.products.insert_one(doc)
    doc.pop("_id", None)
    return doc


@api.put("/products/{product_id}")
async def update_product(product_id: str, body: ProductIn, user: dict = Depends(require_permission("products.edit"))):
    upd = body.model_dump()
    upd["updated_at"] = now_vn()
    res = await db.products.update_one({"product_id": product_id}, {"$set": upd})
    if not res.matched_count:
        raise HTTPException(status_code=404, detail="Không tìm thấy sản phẩm")
    return await db.products.find_one({"product_id": product_id}, {"_id": 0})


@api.delete("/products/{product_id}")
async def delete_product(product_id: str, user: dict = Depends(require_permission("products.delete"))):
    res = await db.products.delete_one({"product_id": product_id})
    if not res.deleted_count:
        raise HTTPException(status_code=404, detail="Không tìm thấy sản phẩm")
    return {"ok": True}


# ===========================================================================
# Raw Materials (NVL)
# ===========================================================================
@api.get("/materials")
async def list_materials(
    q: Optional[str] = None,
    low_stock: bool = False,
    sort: str = "created_at",
    direction: Literal["asc", "desc"] = "desc",
    user: dict = Depends(require_permission("materials.view")),
):
    query = {}
    if q:
        query["$or"] = [{"name": {"$regex": q, "$options": "i"}}, {"code": {"$regex": q, "$options": "i"}}]
    if low_stock:
        query["$expr"] = {"$lte": ["$stock", "$low_stock_threshold"]}
    dirv = 1 if direction == "asc" else -1
    items = await db.materials.find(query, {"_id": 0}).sort(sort, dirv).limit(500).to_list(None)
    return items


@api.post("/materials")
async def create_material(body: MaterialIn, user: dict = Depends(require_permission("materials.create"))):
    doc = body.model_dump()
    doc["material_id"] = f"mat_{uuid.uuid4().hex[:10]}"
    doc["code"] = doc.get("code") or doc["material_id"].upper()
    doc["created_at"] = now_vn()
    doc["updated_at"] = doc["created_at"]
    await db.materials.insert_one(doc)
    doc.pop("_id", None)
    return doc


@api.put("/materials/{material_id}")
async def update_material(material_id: str, body: MaterialIn, user: dict = Depends(require_permission("materials.edit"))):
    upd = body.model_dump()
    upd["updated_at"] = now_vn()
    res = await db.materials.update_one({"material_id": material_id}, {"$set": upd})
    if not res.matched_count:
        raise HTTPException(status_code=404, detail="Không tìm thấy NVL")
    return await db.materials.find_one({"material_id": material_id}, {"_id": 0})


@api.delete("/materials/{material_id}")
async def delete_material(material_id: str, user: dict = Depends(require_permission("materials.delete"))):
    res = await db.materials.delete_one({"material_id": material_id})
    if not res.deleted_count:
        raise HTTPException(status_code=404, detail="Không tìm thấy NVL")
    return {"ok": True}


# ===========================================================================
# Stock movements (in / out / adjust) - FIFO + Expiration
# ===========================================================================
@api.post("/stock/in")
async def stock_in(body: StockInIn, user: dict = Depends(require_permission("stock.in"))):
    now = now_vn()
    coll = "materials" if body.kind == "material" else "products"
    id_field = "material_id" if body.kind == "material" else "product_id"
    target = await db[coll].find_one({id_field: body.target_id})
    if not target:
        raise HTTPException(status_code=404, detail="Không tìm thấy mục nhập kho")

    # parse dates
    prod_date = None
    exp_date = None
    try:
        if body.production_date:
            prod_date = datetime.fromisoformat(body.production_date).replace(tzinfo=timezone.utc)
        if body.expiration_date:
            exp_date = datetime.fromisoformat(body.expiration_date).replace(tzinfo=timezone.utc)
        elif prod_date:
            # Default shelf life from target (default 30d for materials, 3d for products)
            default_days = int(target.get("expiration_days", 30 if body.kind == "material" else 3))
            exp_date = prod_date + timedelta(days=default_days)
    except Exception:
        prod_date = None
        exp_date = None

    movement = {
        "movement_id": f"mov_{uuid.uuid4().hex[:10]}",
        "kind": body.kind, "target_id": body.target_id, "target_name": target.get("name"),
        "type": "in",
        "quantity": body.quantity,
        "remaining": body.quantity,  # for FIFO
        "unit_price": body.unit_price,
        "subtotal": body.unit_price * body.quantity,
        "supplier_id": body.supplier_id, "supplier_name": body.supplier_name,
        "production_date": prod_date,
        "expiration_date": exp_date,
        "batch_code": body.batch_code or f"BATCH-{now.strftime('%y%m%d%H%M')}",
        "notes": body.notes,
        "created_at": now, "created_by": user.get("name", "system"), "created_by_id": user["user_id"],
    }
    await db.stock_movements.insert_one(movement)
    upd = {"$inc": {"stock": body.quantity}}
    if body.kind == "product" and exp_date:
        upd["$set"] = {"latest_expiration_date": exp_date, "latest_production_date": prod_date}
    await db[coll].update_one({id_field: body.target_id}, upd)

    # Auto-create supplier debt when supplier + unit_price > 0 (do NOT log as payment yet)
    if body.supplier_id and body.unit_price > 0:
        debt_total = body.unit_price * body.quantity
        await db.suppliers.update_one(
            {"supplier_id": body.supplier_id},
            {"$inc": {"total_debt": debt_total}},
        )

    movement.pop("_id", None)
    return movement


@api.post("/stock/adjust")
async def stock_adjust(body: StockAdjustIn, user: dict = Depends(require_permission("stock.adjust"))):
    now = now_vn()
    coll = "materials" if body.kind == "material" else "products"
    id_field = "material_id" if body.kind == "material" else "product_id"
    target = await db[coll].find_one({id_field: body.target_id})
    if not target:
        raise HTTPException(status_code=404, detail="Không tìm thấy")

    movement = {
        "movement_id": f"mov_{uuid.uuid4().hex[:10]}",
        "kind": body.kind, "target_id": body.target_id, "target_name": target.get("name"),
        "type": "adjust",
        "quantity": body.delta,
        "reason": body.reason,
        "notes": body.notes,
        "created_at": now, "created_by": user.get("name", "system"), "created_by_id": user["user_id"],
    }
    await db.stock_movements.insert_one(movement)
    await db[coll].update_one({id_field: body.target_id}, {"$inc": {"stock": body.delta}})
    movement.pop("_id", None)
    return movement


@api.get("/stock/movements")
async def list_movements(
    kind: Optional[str] = None, target_id: Optional[str] = None,
    type: Optional[str] = None,
    date_from: Optional[str] = None, date_to: Optional[str] = None,
    limit: int = 500,
    user: dict = Depends(get_current_user),
):
    q = {}
    if kind:
        q["kind"] = kind
    if target_id:
        q["target_id"] = target_id
    if type:
        q["type"] = type
    if date_from or date_to:
        rng = {}
        if date_from:
            rng["$gte"] = datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc)
        if date_to:
            rng["$lte"] = (datetime.fromisoformat(date_to) + timedelta(days=1)).replace(tzinfo=timezone.utc)
        q["created_at"] = rng
    items = await db.stock_movements.find(q, {"_id": 0}).sort("created_at", -1).limit(min(limit, 5000)).to_list(None)
    # Add quantity_change for frontend compatibility
    for item in items:
        qty = item.get("quantity", 0)
        if item.get("type") == "in":
            item["quantity_change"] = abs(qty)
        elif item.get("type") == "out":
            item["quantity_change"] = -abs(qty)
        else:
            item["quantity_change"] = qty
    return items


@api.get("/inventory/xnt")
async def inventory_xnt(
    kind: str = "product",  # "product" | "material"
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    user: dict = Depends(require_permission("reports.view")),
):
    """Xuất Nhập Tồn report — for each entity, returns opening_stock, total_in, total_out,
    ending_stock and current_stock (DB snapshot)."""
    if kind not in ("product", "material"):
        raise HTTPException(status_code=400, detail="kind must be product or material")
    coll = db.products if kind == "product" else db.materials
    id_field = "product_id" if kind == "product" else "material_id"

    df = datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc) if date_from else None
    dt = (datetime.fromisoformat(date_to) + timedelta(days=1)).replace(tzinfo=timezone.utc) if date_to else None

    entities = await coll.find({}, {"_id": 0}).to_list(None)

    rows = []
    for ent in entities:
        eid = ent[id_field]
        # Movements before date_from = opening stock movements
        movs_query = {"kind": kind, "target_id": eid}
        all_movs = await db.stock_movements.find(movs_query, {"_id": 0}).sort("created_at", 1).to_list(None)

        opening = 0
        total_in = 0
        total_out = 0
        for m in all_movs:
            ts = m.get("created_at")
            if ts and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            raw_qty = m.get("quantity", 0) or 0
            mov_type = m.get("type", "")
            if mov_type == "in":
                qty = abs(raw_qty)
            elif mov_type == "out":
                qty = -abs(raw_qty)
            else:
                qty = raw_qty  # adjust hoac legacy
            if df and ts and ts < df:
                opening += qty
                continue
            if dt and ts and ts >= dt:
                continue
            if qty > 0:
                total_in += qty
            else:
                total_out += -qty

        ending = opening + total_in - total_out
        rows.append({
            "id": eid,
            "name": ent.get("name"),
            "unit": ent.get("unit") or ("cái" if kind == "product" else "kg"),
            "opening_stock": opening,
            "total_in": total_in,
            "total_out": total_out,
            "ending_stock": ending,
            "current_stock": ent.get("stock", 0),
        })
    rows.sort(key=lambda r: (r["total_in"] + r["total_out"]), reverse=True)
    totals = {
        "total_in": sum(r["total_in"] for r in rows),
        "total_out": sum(r["total_out"] for r in rows),
        "total_opening": sum(r["opening_stock"] for r in rows),
        "total_ending": sum(r["ending_stock"] for r in rows),
        "n_entities": len(rows),
    }
    return {"items": rows, "totals": totals, "kind": kind, "date_from": date_from, "date_to": date_to}


@api.get("/stock/expiring")
async def expiring_soon(days: int = 7, user: dict = Depends(get_current_user)):
    """Materials/products expiring within N days."""
    now = now_vn()
    cutoff = now + timedelta(days=days)
    items = await db.stock_movements.find(
        {"type": "in", "expiration_date": {"$ne": None, "$lte": cutoff}, "remaining": {"$gt": 0}},
        {"_id": 0},
    ).sort("expiration_date", 1).limit(200).to_list(None)
    return items


# ===========================================================================
# Phase 5 — Production (Sản xuất)
# ===========================================================================
class ProductionBatchIn(BaseModel):
    product_id: str
    quantity: int
    shift: Literal["morning", "afternoon", "evening", "night"] = "morning"
    production_date: Optional[str] = None  # YYYY-MM-DD
    expiration_date: Optional[str] = None
    batch_code: Optional[str] = None
    materials_used: List[dict] = []  # [{material_id, quantity}]
    notes: Optional[str] = ""


@api.get("/production")
async def list_production(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    shift: Optional[str] = None,
    user: dict = Depends(get_current_user),
):
    q = {}
    if shift:
        q["shift"] = shift
    date_q = {}
    if date_from:
        date_q["$gte"] = datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc)
    if date_to:
        date_q["$lte"] = datetime.fromisoformat(date_to).replace(tzinfo=timezone.utc) + timedelta(days=1)
    if date_q:
        q["production_date"] = date_q
    items = await db.production_batches.find(q, {"_id": 0}).sort("production_date", -1).limit(500).to_list(None)
    return items


@api.post("/production")
async def create_production(body: ProductionBatchIn, user: dict = Depends(require_permission("stock.in"))):
    now = now_vn()
    product = await db.products.find_one({"product_id": body.product_id})
    if not product:
        raise HTTPException(status_code=404, detail="Không tìm thấy sản phẩm")

    prod_date = datetime.fromisoformat(body.production_date).replace(tzinfo=timezone.utc) if body.production_date else now
    exp_date = datetime.fromisoformat(body.expiration_date).replace(tzinfo=timezone.utc) if body.expiration_date else (
        prod_date + timedelta(days=int(product.get("expiration_days", 3)))
    )

    doc = {
        "batch_id": f"pb_{uuid.uuid4().hex[:10]}",
        "batch_code": body.batch_code or f"SX-{prod_date.strftime('%y%m%d')}-{body.shift[:1].upper()}",
        "product_id": body.product_id,
        "product_name": product.get("name"),
        "quantity": body.quantity,
        "shift": body.shift,
        "production_date": prod_date,
        "expiration_date": exp_date,
        "materials_used": body.materials_used,
        "notes": body.notes,
        "created_at": now, "created_by": user.get("name"), "created_by_id": user["user_id"],
    }
    await db.production_batches.insert_one(doc)
    # Increment product stock + record stock movement
    await db.products.update_one(
        {"product_id": body.product_id},
        {"$inc": {"stock": body.quantity}, "$set": {"latest_expiration_date": exp_date, "latest_production_date": prod_date}},
    )
    await db.stock_movements.insert_one({
        "movement_id": f"mov_{uuid.uuid4().hex[:10]}",
        "kind": "product",
        "target_id": body.product_id,
        "target_name": product.get("name"),
        "type": "production",
        "quantity": body.quantity,
        "remaining": body.quantity,
        "production_date": prod_date,
        "expiration_date": exp_date,
        "batch_code": doc["batch_code"],
        "notes": f"Sản xuất ca {body.shift}",
        "created_at": now, "created_by": user.get("name"), "created_by_id": user["user_id"],
    })
    # Decrement materials used
    for mu in body.materials_used:
        if mu.get("material_id") and mu.get("quantity"):
            await db.materials.update_one(
                {"material_id": mu["material_id"]}, {"$inc": {"stock": -float(mu["quantity"])}},
            )
    doc.pop("_id", None)
    return doc


@api.delete("/production/{batch_id}")
async def delete_production(batch_id: str, user: dict = Depends(require_permission("stock.adjust"))):
    b = await db.production_batches.find_one({"batch_id": batch_id})
    if not b:
        raise HTTPException(status_code=404, detail="Không tìm thấy lô")
    # rollback stock
    await db.products.update_one({"product_id": b["product_id"]}, {"$inc": {"stock": -b["quantity"]}})
    for mu in b.get("materials_used", []):
        if mu.get("material_id") and mu.get("quantity"):
            await db.materials.update_one({"material_id": mu["material_id"]}, {"$inc": {"stock": float(mu["quantity"])}})
    await db.production_batches.delete_one({"batch_id": batch_id})
    return {"ok": True}


@api.get("/production/forecast")
async def production_forecast(days_ahead: int = 1, user: dict = Depends(get_current_user)):
    """AI-powered recommendation for tomorrow's production quantity per product."""
    now = now_vn()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    target_date = today_start + timedelta(days=days_ahead)
    target_dow = target_date.weekday()  # 0=Monday

    # Look back 28 days for sold quantities per product per day-of-week
    cutoff = today_start - timedelta(days=28)
    pipeline = [
        {"$match": {"created_at": {"$gte": cutoff}, "status": {"$ne": "cancelled"}}},
        {"$unwind": "$items"},
        {"$project": {
            "items": 1,
            "dow": {"$mod": [{"$add": [{"$dayOfWeek": "$created_at"}, 5]}, 7]},  # Mongo: Sun=1; offset to Mon=0
            "day": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
        }},
        {"$group": {
            "_id": {"product_id": "$items.product_id", "name": "$items.name", "dow": "$dow", "day": "$day"},
            "qty": {"$sum": "$items.quantity"},
        }},
    ]
    daily_buckets = []
    async for r in db.orders.aggregate(pipeline):
        daily_buckets.append(r)

    # Aggregate per (product, dow): list of daily quantities
    by_prod_dow = {}
    for r in daily_buckets:
        key = (r["_id"]["product_id"], r["_id"]["dow"])
        by_prod_dow.setdefault(key, {"name": r["_id"]["name"], "values": []})
        by_prod_dow[key]["values"].append(r["qty"])

    forecasts = []
    for (pid, dow), info in by_prod_dow.items():
        if dow != target_dow:
            continue
        values = info["values"]
        if not values:
            continue
        avg = sum(values) / len(values)
        recommended = int(round(avg * 1.1))  # +10% buffer
        product = await db.products.find_one({"product_id": pid}, {"_id": 0, "stock": 1, "low_stock_threshold": 1, "unit": 1, "name": 1})
        current_stock = product.get("stock", 0) if product else 0
        needed = max(0, recommended - current_stock)
        forecasts.append({
            "product_id": pid,
            "name": info["name"],
            "day_of_week": dow,
            "avg_daily_sold": round(avg, 1),
            "samples": len(values),
            "recommended_production": recommended,
            "current_stock": current_stock,
            "needs_to_produce": needed,
            "unit": product.get("unit", "cái") if product else "cái",
        })
    forecasts.sort(key=lambda x: x["needs_to_produce"], reverse=True)

    # Aggregate summary for AI suggestion text
    DOW_NAMES = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"]
    return {
        "target_date": target_date.strftime("%Y-%m-%d"),
        "target_day_of_week": DOW_NAMES[target_dow],
        "lookback_days": 28,
        "forecasts": forecasts,
    }


# ===========================================================================
# Phase 8 — Smart insights (Combo + Customer scoring + Churn)
# ===========================================================================
@api.get("/insights/combos")
async def insight_combos(min_count: int = 2, user: dict = Depends(get_current_user)):
    """Frequently bought together: pairs of products that appear in the same order."""
    pipeline = [
        {"$match": {"status": {"$ne": "cancelled"}}},
        {"$project": {"product_ids": "$items.product_id", "items": 1}},
        {"$match": {"$expr": {"$gte": [{"$size": "$product_ids"}, 2]}}},
        {"$limit": 5000},
    ]
    pairs_count: Dict[str, Dict[str, Any]] = {}
    async for o in db.orders.aggregate(pipeline):
        items = o.get("items", [])
        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                a = items[i]
                b = items[j]
                if a.get("product_id") == b.get("product_id"):
                    continue
                key = tuple(sorted([a.get("product_id", ""), b.get("product_id", "")]))
                key_str = "|".join(key)
                if key_str not in pairs_count:
                    name_a = a["name"] if a.get("product_id") == key[0] else b["name"]
                    name_b = b["name"] if b.get("product_id") == key[1] else a["name"]
                    pairs_count[key_str] = {
                        "product_a_id": key[0], "product_b_id": key[1],
                        "name_a": name_a, "name_b": name_b,
                        "count": 0, "revenue": 0.0,
                    }
                pairs_count[key_str]["count"] += 1
                pairs_count[key_str]["revenue"] += a.get("subtotal", 0) + b.get("subtotal", 0)

    filtered = [v for v in pairs_count.values() if v["count"] >= min_count]
    filtered.sort(key=lambda x: x["count"], reverse=True)
    return {"items": filtered[:30]}


@api.get("/insights/customer-scoring")
async def insight_customer_scoring(user: dict = Depends(get_current_user)):
    """Rule-based scoring: classify customers as VIP / High potential / At risk / Churned."""
    now = now_vn()
    customers = await db.customers.find({}, {"_id": 0}).limit(2000).to_list(None)
    pipeline = [
        {"$match": {"status": {"$ne": "cancelled"}, "customer_id": {"$ne": None}}},
        {"$group": {
            "_id": "$customer_id",
            "last_order": {"$max": "$created_at"},
            "first_order": {"$min": "$created_at"},
            "order_count": {"$sum": 1},
            "total_spent": {"$sum": "$total"},
            "avg_order": {"$avg": "$total"},
        }},
    ]
    stats = {}
    async for r in db.orders.aggregate(pipeline):
        stats[r["_id"]] = r

    out = {"vip": [], "high_potential": [], "at_risk": [], "churned": [], "new": []}
    for c in customers:
        s = stats.get(c["customer_id"], {})
        last_order = _ensure_dt(s.get("last_order"))
        order_count = s.get("order_count", 0)
        total_spent = s.get("total_spent", 0)
        avg_order = s.get("avg_order", 0)
        days_since = (now - last_order).days if last_order else 9999

        scored = {
            **c,
            "order_count": order_count,
            "total_spent": total_spent,
            "avg_order_value": round(avg_order, 0),
            "days_since_last_order": days_since,
        }

        if order_count == 0:
            out["new"].append(scored)
        elif total_spent >= 1_000_000 and order_count >= 5 and days_since <= 14:
            out["vip"].append(scored)
        elif days_since >= 30 and order_count >= 2:
            out["at_risk"].append(scored)
        elif days_since >= 60:
            out["churned"].append(scored)
        elif order_count >= 2 and avg_order >= 100_000:
            out["high_potential"].append(scored)
        else:
            out["new"].append(scored)

    return {
        "counts": {k: len(v) for k, v in out.items()},
        "vip": out["vip"][:50],
        "high_potential": out["high_potential"][:50],
        "at_risk": out["at_risk"][:50],
        "churned": out["churned"][:50],
        "new": out["new"][:50],
    }


@api.get("/insights/ai-suggest")
async def insight_ai_suggest(user: dict = Depends(get_current_user)):
    """Use LLM to generate smart marketing/operational suggestions."""
    if not EMERGENT_LLM_KEY:
        raise HTTPException(status_code=500, detail="LLM key chưa cấu hình")

    # Gather context
    scoring = await insight_customer_scoring(user)
    combos = await insight_combos(min_count=2, user=user)
    forecast = await production_forecast(days_ahead=1, user=user)

    context = f"""DỮ LIỆU NỘI BỘ:
- Khách hàng VIP: {scoring['counts']['vip']}
- Khách có tiềm năng: {scoring['counts']['high_potential']}
- Khách nguy cơ rời bỏ: {scoring['counts']['at_risk']}
- Khách đã rời: {scoring['counts']['churned']}
- Khách mới: {scoring['counts']['new']}

5 cặp sản phẩm hay mua chung:
{chr(10).join([f"  - {c['name_a']} + {c['name_b']}: {c['count']} đơn" for c in combos['items'][:5]]) or '  (chưa đủ data)'}

Top 5 sản phẩm cần sản xuất ngày mai ({forecast['target_day_of_week']}):
{chr(10).join([f"  - {f['name']}: gợi ý {f['recommended_production']} {f['unit']}" for f in forecast['forecasts'][:5]]) or '  (chưa đủ data)'}
"""

    system = """Bạn là chuyên gia kinh doanh F&B. Phân tích dữ liệu và đưa ra 3-5 gợi ý hành động NGAY tuần này.
Định dạng JSON đơn giản:
{
  "summary": "1-2 câu tổng quan",
  "actions": [
    {"title": "...", "detail": "...", "priority": "high|medium|low"}
  ]
}
Chỉ trả về JSON, không markdown."""

    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"insights_{user['user_id']}_{now_vn().strftime('%Y%m%d')}",
            system_message=system,
        ).with_model("anthropic", "claude-sonnet-4-5-20250929")
        response = await chat.send_message(UserMessage(text=context))
        # Try parse JSON from response
        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("```")[1].lstrip("json").strip()
            parsed = json.loads(text)
            return {"raw": response, "parsed": parsed}
        except Exception:
            return {"raw": response, "parsed": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)[:200]}")


# ===========================================================================
# Customers
# ===========================================================================
@api.get("/customers")
async def list_customers(
    q: Optional[str] = None,
    group: Optional[str] = None,
    type: Optional[str] = None,
    district: Optional[str] = None,
    classification: Optional[str] = None,
    sort: str = "created_at",
    direction: Literal["asc", "desc"] = "desc",
    user: dict = Depends(require_permission("customers.view")),
):
    query = {}
    if q:
        query["$or"] = [
            {"name": {"$regex": q, "$options": "i"}},
            {"nickname": {"$regex": q, "$options": "i"}},
            {"phone": {"$regex": q, "$options": "i"}},
            {"email": {"$regex": q, "$options": "i"}},
            {"code": {"$regex": q, "$options": "i"}},
        ]
    if group:
        query["group"] = group
    if type:
        query["type"] = type
    if district:
        query["district"] = district
    if classification:
        query["classification"] = classification
    dirv = 1 if direction == "asc" else -1
    items = await db.customers.find(query, {"_id": 0}).sort(sort, dirv).limit(2000).to_list(None)
    return items


async def _gen_customer_code() -> str:
    last = await db.customers.find_one(
        {"code": {"$regex": "^KH\\d+$"}},
        sort=[("code", -1)]
    )
    if last and last.get("code"):
        try:
            num = int(last["code"][2:]) + 1
        except Exception:
            num = 1
    else:
        num = 1
    return f"KH{str(num).zfill(4)}"


@api.post("/customers")
async def create_customer(body: CustomerIn, user: dict = Depends(require_permission("customers.create"))):
    doc = body.model_dump()
    doc["customer_id"] = f"cus_{uuid.uuid4().hex[:10]}"
    doc["code"] = doc.get("code") or await _gen_customer_code()
    doc["created_at"] = now_vn()
    doc["total_orders"] = 0
    doc["total_spent"] = 0.0
    doc["last_order_at"] = None
    await db.customers.insert_one(doc)
    doc.pop("_id", None)
    return doc


@api.put("/customers/{customer_id}")
async def update_customer(customer_id: str, body: CustomerIn, user: dict = Depends(require_permission("customers.edit"))):
    upd = body.model_dump()
    res = await db.customers.update_one({"customer_id": customer_id}, {"$set": upd})
    if not res.matched_count:
        raise HTTPException(status_code=404, detail="Không tìm thấy khách hàng")
    return await db.customers.find_one({"customer_id": customer_id}, {"_id": 0})


@api.delete("/customers/{customer_id}")
async def delete_customer(customer_id: str, user: dict = Depends(require_permission("customers.delete"))):
    res = await db.customers.delete_one({"customer_id": customer_id})
    if not res.deleted_count:
        raise HTTPException(status_code=404, detail="Không tìm thấy")
    return {"ok": True}


@api.get("/customers/care")
async def customer_care(days: int = 14, user: dict = Depends(require_permission("customers.view"))):
    """days=0 means 'has not ordered TODAY'. days>0 means hasn't ordered in N+ days."""
    now = now_vn()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    cutoff = today_start if days == 0 else (now - timedelta(days=days))
    customers = await db.customers.find({}, {"_id": 0}).limit(2000).to_list(None)

    pipeline = [
        {"$match": {"customer_id": {"$ne": None}, "status": {"$ne": "cancelled"}}},
        {"$group": {
            "_id": "$customer_id",
            "last_order": {"$max": "$created_at"},
            "order_count": {"$sum": 1},
            "total_spent": {"$sum": "$total"},
        }},
    ]
    last_orders = {}
    async for r in db.orders.aggregate(pipeline):
        last_orders[r["_id"]] = r

    # Today + month aggregates
    today_pipeline = [
        {"$match": {"created_at": {"$gte": today_start}, "status": {"$ne": "cancelled"}, "customer_id": {"$ne": None}}},
        {"$group": {"_id": "$customer_id", "today_orders": {"$sum": 1}, "today_spent": {"$sum": "$total"}}},
    ]
    today_stats = {}
    async for r in db.orders.aggregate(today_pipeline):
        today_stats[r["_id"]] = r

    month_pipeline = [
        {"$match": {"created_at": {"$gte": month_start}, "status": {"$ne": "cancelled"}, "customer_id": {"$ne": None}}},
        {"$group": {"_id": "$customer_id", "month_orders": {"$sum": 1}, "month_spent": {"$sum": "$total"}}},
    ]
    month_stats = {}
    async for r in db.orders.aggregate(month_pipeline):
        month_stats[r["_id"]] = r

    enriched = []
    for c in customers:
        lo = last_orders.get(c["customer_id"], {})
        ts = today_stats.get(c["customer_id"], {})
        ms = month_stats.get(c["customer_id"], {})
        last_order = _ensure_dt(lo.get("last_order"))
        if days == 0:
            needs_care = ts.get("today_orders", 0) == 0
        else:
            needs_care = (last_order is None) or (last_order and last_order < cutoff)
        enriched.append({
            **c,
            "last_order": last_order,
            "order_count": lo.get("order_count", 0),
            "total_spent_agg": lo.get("total_spent", 0),
            "today_orders": ts.get("today_orders", 0),
            "today_spent": ts.get("today_spent", 0),
            "month_orders": ms.get("month_orders", 0),
            "month_spent": ms.get("month_spent", 0),
            "needs_care": needs_care,
        })

    enriched.sort(key=lambda x: (x["last_order"] is None, -(x["last_order"].timestamp() if x["last_order"] else 0)))

    return {
        "items": enriched,
        "total": len(enriched),
        "needs_care_count": sum(1 for x in enriched if x["needs_care"]),
        "cutoff_days": days,
    }


@api.get("/customers/order-stats")
async def customer_order_stats(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    user: dict = Depends(require_permission("customers.view")),
):
    """Thong ke don hang theo customer_id cho khoang thoi gian cho truoc."""
    match = {"status": {"$ne": "cancelled"}, "customer_id": {"$ne": None}}
    if date_from or date_to:
        date_q = {}
        if date_from:
            try:
                dt = datetime.fromisoformat(date_from)
                if dt.tzinfo is None:
                    dt = dt.replace(hour=0, minute=0, second=0, tzinfo=VN_TZ)
                date_q["$gte"] = dt
            except Exception:
                pass
        if date_to:
            try:
                dt = datetime.fromisoformat(date_to)
                if dt.tzinfo is None:
                    dt = dt.replace(hour=23, minute=59, second=59, tzinfo=VN_TZ)
                date_q["$lte"] = dt
            except Exception:
                pass
        if date_q:
            match["created_at"] = date_q

    pipeline = [
        {"$match": match},
        {"$group": {
            "_id": "$customer_id",
            "orders": {"$sum": 1},
            "spent": {"$sum": "$total"},
        }},
    ]
    result = {}
    async for r in db.orders.aggregate(pipeline):
        result[r["_id"]] = {"orders": r["orders"], "spent": r["spent"]}
    return result


@api.get("/customers/export")
async def export_customers(user: dict = Depends(require_permission("customers.view"))):
    """Export all customers to Excel."""
    customers = await db.customers.find({}, {"_id": 0}).limit(5000).to_list(None)
    df = pd.DataFrame(customers if customers else [{}])
    # Reorder columns for readability
    preferred = ["code", "name", "nickname", "phone", "email", "address", "district", "city",
                 "tax_id", "group", "type", "classification", "max_debt_days", "max_debt_amount",
                 "total_orders", "total_spent", "notes"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred and c not in ("customer_id", "created_at", "assigned_user_id")]
    if cols:
        df = df[cols]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="customers")
    buf.seek(0)
    fname = f"customers-{datetime.now().strftime('%Y%m%d-%H%M')}.xlsx"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@api.get("/customers/import-template")
async def customer_import_template(user: dict = Depends(require_permission("customers.import"))):
    """Download Excel template for customer import."""
    sample = [{
        "code": "KH001",
        "name": "Nguyễn Văn A",
        "nickname": "Anh A",
        "phone": "0901234567",
        "email": "a@email.com",
        "address": "123 Lê Lợi",
        "district": "Quận 1",
        "city": "TP.HCM",
        "tax_id": "",
        "group": "regular",
        "type": "retail",
        "classification": "",
        "max_debt_days": 0,
        "max_debt_amount": 0,
        "notes": "",
    }]
    df = pd.DataFrame(sample)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="customers")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="customers-template.xlsx"'},
    )


@api.post("/customers/import")
async def import_customers(file: UploadFile = File(...), user: dict = Depends(require_permission("customers.import"))):
    raw = await file.read()
    try:
        df = pd.read_excel(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File không đọc được: {e}")
    inserted = 0
    skipped = 0
    errors: List[str] = []
    now = now_vn()
    for idx, row in df.iterrows():
        try:
            name = str(row.get("name", "")).strip()
            if not name or name.lower() == "nan":
                skipped += 1
                continue
            code_v = row.get("code")
            phone_v = row.get("phone")
            doc = {
                "customer_id": f"cus_{uuid.uuid4().hex[:10]}",
                "code": str(code_v).strip() if pd.notna(code_v) and str(code_v).strip() else await _gen_customer_code(),
                "name": name,
                "nickname": str(row.get("nickname") or "").strip(),
                "phone": str(phone_v).strip() if pd.notna(phone_v) else "",
                "email": str(row.get("email") or "").strip(),
                "address": str(row.get("address") or "").strip(),
                "district": str(row.get("district") or "").strip(),
                "city": str(row.get("city") or "").strip(),
                "tax_id": str(row.get("tax_id") or "").strip(),
                "group": str(row.get("group") or "new").strip().lower(),
                "type": str(row.get("type") or "retail").strip().lower(),
                "classification": str(row.get("classification") or "").strip(),
                "assigned_user_id": "",
                "max_debt_days": int(float(row.get("max_debt_days") or 0)),
                "max_debt_amount": float(row.get("max_debt_amount") or 0),
                "notes": str(row.get("notes") or "").strip(),
                "total_orders": 0,
                "total_spent": 0.0,
                "last_order_at": None,
                "created_at": now,
            }
            # Skip if a customer with same name+phone already exists
            existing = await db.customers.find_one({"name": doc["name"], "phone": doc["phone"]})
            if existing:
                skipped += 1
                continue
            await db.customers.insert_one(doc)
            inserted += 1
        except Exception as e:
            errors.append(f"Dòng {idx + 2}: {e}")
    return {"inserted": inserted, "skipped": skipped, "errors": errors[:10]}


# ===========================================================================
# Suppliers
# ===========================================================================
@api.get("/suppliers")
async def list_suppliers(
    q: Optional[str] = None, group: Optional[str] = None,
    sort: str = "created_at", direction: Literal["asc", "desc"] = "desc",
    user: dict = Depends(require_permission("suppliers.view")),
):
    query = {}
    if q:
        query["$or"] = [{"name": {"$regex": q, "$options": "i"}}, {"phone": {"$regex": q, "$options": "i"}}]
    if group:
        query["group"] = group
    dirv = 1 if direction == "asc" else -1
    items = await db.suppliers.find(query, {"_id": 0}).sort(sort, dirv).limit(500).to_list(None)
    return items


@api.post("/suppliers")
async def create_supplier(body: SupplierIn, user: dict = Depends(require_permission("suppliers.create"))):
    doc = body.model_dump()
    doc["supplier_id"] = f"sup_{uuid.uuid4().hex[:10]}"
    doc["code"] = doc.get("code") or f"NCC{now_vn().strftime('%y%m')}{uuid.uuid4().hex[:4].upper()}"
    doc["created_at"] = now_vn()
    doc["total_debt"] = 0
    await db.suppliers.insert_one(doc)
    doc.pop("_id", None)
    return doc


@api.put("/suppliers/{supplier_id}")
async def update_supplier(supplier_id: str, body: SupplierIn, user: dict = Depends(require_permission("suppliers.edit"))):
    upd = body.model_dump()
    res = await db.suppliers.update_one({"supplier_id": supplier_id}, {"$set": upd})
    if not res.matched_count:
        raise HTTPException(status_code=404, detail="Không tìm thấy NCC")
    return await db.suppliers.find_one({"supplier_id": supplier_id}, {"_id": 0})


@api.delete("/suppliers/{supplier_id}")
async def delete_supplier(supplier_id: str, user: dict = Depends(require_permission("suppliers.delete"))):
    res = await db.suppliers.delete_one({"supplier_id": supplier_id})
    if not res.deleted_count:
        raise HTTPException(status_code=404, detail="Không tìm thấy NCC")
    return {"ok": True}


# ===========================================================================
# Orders
# ===========================================================================
async def _generate_order_code() -> str:
    year = now_vn().strftime('%y')
    prefix = f"BB{year}"
    last = await db.orders.find_one(
        {"order_code": {"$regex": f"^{prefix}\\d+$"}},
        sort=[("order_code", -1)]
    )
    if last and last.get("order_code"):
        try:
            num = int(last["order_code"][4:]) + 1
        except Exception:
            num = 1
    else:
        num = 1
    return f"{prefix}{str(num).zfill(6)}"


@api.get("/orders")
async def list_orders(
    q: Optional[str] = None,
    customer_id: Optional[str] = None,
    status: Optional[str] = None,
    payment_method: Optional[str] = None,
    type: Optional[str] = None,
    district: Optional[str] = None,
    shipper_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sort: str = "created_at",
    direction: Literal["asc", "desc"] = "desc",
    user: dict = Depends(require_permission("orders.view")),
):
    query = {}
    if q:
        query["$or"] = [
            {"order_code": {"$regex": q, "$options": "i"}},
            {"customer_name": {"$regex": q, "$options": "i"}},
            {"customer_phone": {"$regex": q, "$options": "i"}},
        ]
    
    # Logic lọc mới
    if customer_id:
        query["customer_id"] = customer_id
    if status:
        query["status"] = status
    if payment_method:
        query["payment_method"] = payment_method
    if type:
        query["type"] = type
    if district:
        query["customer_district"] = district
    if shipper_id:
        query["assigned_shipper_id"] = shipper_id

    # Xử lý thời gian — date_to cộng thêm 1 ngày để bao gồm toàn bộ ngày đó
    date_q = {}
    if date_from:
        date_q["$gte"] = datetime.fromisoformat(date_from)
    if date_to:
        date_q["$lt"] = datetime.fromisoformat(date_to) + timedelta(days=1)
    if date_q:
        query["created_at"] = date_q

    direc = -1 if direction == "desc" else 1
    cursor = db.orders.find(query, {"_id": 0}).sort(sort, direc).limit(1000)
    return await cursor.to_list(None)

@api.get("/orders/{order_id}")
async def get_order(order_id: str, user: dict = Depends(require_permission("orders.view"))):
    o = await db.orders.find_one({"order_id": order_id}, {"_id": 0})
    if not o:
        raise HTTPException(status_code=404, detail="Không tìm thấy đơn hàng")
    return o


@api.post("/orders")
async def create_order(body: OrderIn, user: dict = Depends(require_permission("orders.create"))):
    items = [it.model_dump() for it in body.items]
    subtotal = sum(it["subtotal"] for it in items)
    percent_discount = round(subtotal * (body.discount_percent or 0) / 100, 2)
    total = subtotal - body.discount - percent_discount + body.shipping_fee

    payment = _normalize_payment(body.payment_method)

    # Neu co so tien con no (thanh toan 1 phan), chuyen sang debt
    has_remaining = body.remaining_amount and body.remaining_amount > 0
    if has_remaining and payment != "debt":
        payment = "debt"  # chuyen thanh cong no

    due_date = None
    if body.due_date:
        try:
            due_date = datetime.fromisoformat(body.due_date).replace(tzinfo=timezone.utc)
        except Exception:
            due_date = None
    # If debt + customer has max_debt_days set, compute default due date
    if not due_date and payment == "debt" and body.customer_id:
        c = await db.customers.find_one({"customer_id": body.customer_id})
        if c and c.get("max_debt_days"):
            due_date = now_vn() + timedelta(days=int(c["max_debt_days"]))

    order_id = f"ord_{uuid.uuid4().hex[:12]}"
    now = now_vn()
    doc = {
        "order_id": order_id,
        "order_code": await _generate_order_code(),
        "customer_id": body.customer_id,
        "customer_name": body.customer_name,
        "customer_phone": body.customer_phone,
        "customer_address": body.customer_address,
        "customer_district": body.customer_district,
        "customer_city": body.customer_city,
        "type": body.type,
        "items": items,
        "subtotal": subtotal,
        "discount": body.discount,
        "discount_percent": body.discount_percent,
        "discount_amount": body.discount + percent_discount,
        "shipping_fee": body.shipping_fee,
        "total": total,
        "paid_amount": total - body.remaining_amount if body.remaining_amount and body.remaining_amount > 0 else (total if payment != "debt" else 0),
        "remaining_amount": body.remaining_amount if body.remaining_amount and body.remaining_amount > 0 else (total if payment == "debt" else 0),
        "payment_method": payment,
        "is_paid": (payment in ("cash", "transfer", "card", "ewallet")) and (not body.remaining_amount or body.remaining_amount <= 0),
        "status": "new",
        "due_date": due_date,
        "assigned_shipper_id": body.assigned_shipper_id or None,
        "note": body.note,
        "timeline": [{
            "status": "new", "at": now,
            "by": user.get("name", "system"), "note": "Tạo đơn hàng mới",
        }],
        "created_at": now, "created_by": user["user_id"], "updated_at": now,
    }
    await db.orders.insert_one(doc)

    # ── Đồng bộ Finance: đơn thanh toán ngay (tiền mặt / chuyển khoản / ...) ─
    paid_amount = doc.get("paid_amount", 0)
    if payment in ("cash", "transfer", "card", "ewallet") and paid_amount > 0:
        vn_tz = timezone(timedelta(hours=7))
        method_label = {"cash": "Tiền mặt", "transfer": "Chuyển khoản", "card": "Thẻ", "ewallet": "Ví điện tử"}.get(payment, payment)
        income_cat = await db.finance_categories.find_one({"type": "income", "name": "Doanh thu bán hàng"}, {"_id": 0})
        await db.finance_transactions.insert_one({
            "transaction_id": f"txn_{uuid.uuid4().hex[:10]}",
            "type": "income",
            "category_id": income_cat["category_id"] if income_cat else None,
            "category_name": "Doanh thu bán hàng",
            "category_color": income_cat["color"] if income_cat else "#5B8A3C",
            "amount": paid_amount,
            "description": f"Đơn {doc['order_code']} - {body.customer_name or ''} ({method_label})",
            "date": now.astimezone(vn_tz).isoformat(),
            "source": "order",
            "ref_id": order_id,
            "created_at": now,
            "created_by": user.get("name"),
            "created_by_id": user["user_id"],
        })
    # ─────────────────────────────────────────────────────────────────────────

    for it in items:
        await db.products.update_one(
            {"product_id": it["product_id"]}, {"$inc": {"stock": -it["quantity"]}},
        )
        # Ghi lich su xuat kho
        await db.stock_movements.insert_one({
            "movement_id": f"mov_{__import__('uuid').uuid4().hex[:10]}",
            "kind": "product",
            "target_id": it["product_id"],
            "target_name": it["name"],
            "type": "out",
            "quantity": -it["quantity"],
            "unit_price": it["price"],
            "order_id": order_id,
            "order_code": doc["order_code"],
            "notes": f"Xuat kho theo don {doc['order_code']}",
            "created_by": user.get("name", "system"),
            "created_by_id": user["user_id"],
            "created_at": now,
        })
    if body.customer_id:
        await db.customers.update_one(
            {"customer_id": body.customer_id},
            {"$inc": {"total_orders": 1, "total_spent": total}, "$set": {"last_order_at": now}},
        )

    doc.pop("_id", None)
    return doc


@api.post("/orders/{order_id}/duplicate")
async def duplicate_order(order_id: str, user: dict = Depends(require_permission("orders.create"))):
    """Copy an existing order and create a new one with current date."""
    src = await db.orders.find_one({"order_id": order_id}, {"_id": 0})
    if not src:
        raise HTTPException(status_code=404, detail="Không tìm thấy đơn hàng gốc")
    now = now_vn()
    new_id = f"ord_{uuid.uuid4().hex[:12]}"
    doc = {
        **src,
        "order_id": new_id,
        "order_code": await _generate_order_code(),
        "status": "new",
        "paid_amount": 0,
        "is_paid": src.get("payment_method") in ("cash", "transfer", "card", "ewallet"),
        "remaining_amount": src.get("total", 0) if src.get("payment_method") == "debt" else 0,
        "timeline": [{
            "status": "new", "at": now,
            "by": user.get("name", "system"),
            "note": f"Nhân bản từ đơn {src.get('order_code')}",
        }],
        "created_at": now, "created_by": user["user_id"], "updated_at": now,
    }
    await db.orders.insert_one(doc)
    # Decrement stock
    for it in doc.get("items", []):
        await db.products.update_one(
            {"product_id": it["product_id"]}, {"$inc": {"stock": -it["quantity"]}},
        )
    if doc.get("customer_id"):
        await db.customers.update_one(
            {"customer_id": doc["customer_id"]},
            {"$inc": {"total_orders": 1, "total_spent": doc.get("total", 0)}, "$set": {"last_order_at": now}},
        )
    doc.pop("_id", None)
    return doc


@api.put("/orders/{order_id}/status")
async def update_order_status(order_id: str, body: OrderStatusUpdate, user: dict = Depends(require_permission("orders.edit"))):
    o = await db.orders.find_one({"order_id": order_id})
    if not o:
        raise HTTPException(status_code=404, detail="Không tìm thấy đơn hàng")
    now = now_vn()
    entry = {"status": body.status, "at": now, "by": user.get("name", "system"), "note": body.note or ""}
    update = {"$set": {"status": body.status, "updated_at": now}, "$push": {"timeline": entry}}
    if body.status == "debt_pending":
        # Khi chuyen sang con no, doi payment_method sang debt va cap nhat remaining_amount
        update["$set"]["payment_method"] = "debt"
        update["$set"]["is_paid"] = False
        total = o.get("total", 0)
        orig_method = o.get("payment_method", "")
        existing_remaining = o.get("remaining_amount") or 0

        if existing_remaining > 0:
            # Đơn đã là nợ 1 phần (vd: nợ từ đầu hoặc đã thu 1 phần qua debts/collect)
            # -> giữ nguyên remaining, không đụng paid_amount
            pass
        elif orig_method in ("cash", "transfer", "card", "ewallet"):
            # Đơn được tạo với TM/CK: paid_amount=total chỉ là giá trị hệ thống tự ghi,
            # khách chưa thực sự trả. Chuyển sang nợ -> toàn bộ là nợ.
            update["$set"]["remaining_amount"] = total
            update["$set"]["paid_amount"] = 0
        else:
            # Đơn nợ từ đầu nhưng remaining=0 (hiếm) -> toàn bộ là nợ
            update["$set"]["remaining_amount"] = total
            update["$set"]["paid_amount"] = 0

        # Xóa finance transaction đã ghi khi tạo đơn (nếu có)
        # Vì khách chưa thực sự trả tiền, không được tính vào doanh thu
        await db.finance_transactions.delete_many({
            "source": "order",
            "ref_id": order_id,
            "type": "income",
        })
    if body.status == "delivered":
        if o.get("payment_method") != "debt":
            update["$set"]["is_paid"] = True
    if body.status == "cancelled":
        for it in o.get("items", []):
            await db.products.update_one(
                {"product_id": it["product_id"]}, {"$inc": {"stock": it["quantity"]}},
            )
    await db.orders.update_one({"order_id": order_id}, update)
    return await db.orders.find_one({"order_id": order_id}, {"_id": 0})


@api.put("/orders/{order_id}/assign-shipper")
async def assign_shipper(order_id: str, shipper_id: str = Query(...), user: dict = Depends(require_permission("delivery.assign"))):
    shipper = await db.users.find_one({"user_id": shipper_id})
    if not shipper:
        raise HTTPException(status_code=404, detail="Shipper không tồn tại")
    res = await db.orders.update_one(
        {"order_id": order_id},
        {"$set": {"assigned_shipper_id": shipper_id, "assigned_shipper_name": shipper.get("name")}},
    )
    if not res.matched_count:
        raise HTTPException(status_code=404, detail="Không tìm thấy đơn hàng")
    return await db.orders.find_one({"order_id": order_id}, {"_id": 0})


@api.delete("/orders/{order_id}")
async def delete_order(order_id: str, user: dict = Depends(require_permission_and_pin2("orders.delete"))):
    o = await db.orders.find_one({"order_id": order_id})
    if not o:
        raise HTTPException(status_code=404, detail="Không tìm thấy")
    if o.get("status") != "cancelled":
        for it in o.get("items", []):
            await db.products.update_one(
                {"product_id": it["product_id"]}, {"$inc": {"stock": it["quantity"]}},
            )
    await db.orders.delete_one({"order_id": order_id})
    return {"ok": True}


# ===========================================================================
# Debts
# ===========================================================================
@api.get("/debts/customers")
async def debts_customers(user: dict = Depends(require_permission("debts.view"))):
    """Aggregated customer debt with overdue/due classification."""
    now = now_vn()
    pipeline = [
        {"$match": {"payment_method": "debt", "is_paid": {"$ne": True}, "status": {"$ne": "cancelled"}}},
        {"$group": {
            "_id": {"customer_id": "$customer_id", "customer_name": "$customer_name"},
            "phone": {"$first": "$customer_phone"},
            "total_debt": {"$sum": "$remaining_amount"},
            "order_count": {"$sum": 1},
            "min_due": {"$min": "$due_date"},
            "max_due": {"$max": "$due_date"},
            "last_order": {"$max": "$created_at"},
        }},
        {"$sort": {"total_debt": -1}},
        {"$limit": 500},
    ]
    items = []
    async for r in db.orders.aggregate(pipeline):
        min_due = _ensure_dt(r.get("min_due"))
        max_due = _ensure_dt(r.get("max_due"))
        overdue = False
        due_soon = False
        if min_due:
            if min_due < now:
                overdue = True
            elif min_due < now + timedelta(days=3):
                due_soon = True
        items.append({
            "customer_id": r["_id"]["customer_id"],
            "customer_name": r["_id"]["customer_name"],
            "phone": r.get("phone"),
            "total_debt": r["total_debt"],
            "order_count": r["order_count"],
            "earliest_due": min_due,
            "latest_due": max_due,
            "last_order": r.get("last_order"),
            "overdue": overdue,
            "due_soon": due_soon,
        })
    total = sum(i["total_debt"] for i in items)
    overdue_total = sum(i["total_debt"] for i in items if i["overdue"])
    due_soon_total = sum(i["total_debt"] for i in items if i["due_soon"])
    return {
        "items": items,
        "total": total,
        "overdue_total": overdue_total,
        "due_soon_total": due_soon_total,
    }


@api.get("/debts/customers/{customer_id}/orders")
async def customer_debt_orders(customer_id: str, user: dict = Depends(require_permission("debts.view"))):
    """All unpaid debt orders for a customer with detail."""
    now = now_vn()
    orders = await db.orders.find(
        {"customer_id": customer_id, "payment_method": "debt", "is_paid": {"$ne": True}, "status": {"$ne": "cancelled"}},
        {"_id": 0},
    ).sort("created_at", -1).limit(200).to_list(None)
    for o in orders:
        due = o.get("due_date")
        if due:
            if isinstance(due, str):
                try:
                    due = datetime.fromisoformat(due.replace("Z", "+00:00"))
                except Exception:
                    due = None
            if due and due.tzinfo is None:
                due = due.replace(tzinfo=timezone.utc)
            if due:
                delta = (due - now).days
                o["days_to_due"] = delta
                o["overdue"] = delta < 0
                o["due_soon"] = 0 <= delta <= 3
    return orders


@api.post("/debts/collect")
async def debts_collect(body: DebtPaymentIn, user: dict = Depends(require_permission("debts.collect"))):
    """Customer pays (partially or fully) on a debt order."""
    if not body.order_id:
        raise HTTPException(status_code=400, detail="order_id is required")
    o = await db.orders.find_one({"order_id": body.order_id})
    if not o:
        raise HTTPException(status_code=404, detail="Không tìm thấy đơn hàng")
    remaining = o.get("remaining_amount", o.get("total", 0)) - body.amount
    is_paid = remaining <= 0
    update = {
        "$set": {
            "remaining_amount": max(0, remaining),
            "paid_amount": o.get("paid_amount", 0) + body.amount,
            "is_paid": is_paid,
        }
    }
    await db.orders.update_one({"order_id": body.order_id}, update)
    now = now_vn()
    payment = {
        "payment_id": f"pmt_{uuid.uuid4().hex[:10]}",
        "direction": "in",  # money coming IN from customer
        "order_id": body.order_id,
        "order_code": o.get("order_code"),
        "customer_id": o.get("customer_id"),
        "customer_name": o.get("customer_name"),
        "amount": body.amount,
        "method": body.method,
        "notes": body.notes,
        "created_at": now, "created_by": user.get("name"), "created_by_id": user["user_id"],
    }
    await db.debt_payments.insert_one(payment)
    payment.pop("_id", None)

    # ── Đồng bộ sang Finance: ghi nhận khoản thu từ công nợ ──────────────────
    vn_tz = timezone(timedelta(hours=7))
    income_cat = await db.finance_categories.find_one({"type": "income", "name": "Thu công nợ"}, {"_id": 0})
    finance_txn = {
        "transaction_id": f"txn_{uuid.uuid4().hex[:10]}",
        "type": "income",
        "category_id": income_cat["category_id"] if income_cat else None,
        "category_name": "Thu công nợ",
        "category_color": income_cat["color"] if income_cat else "#2D4A22",
        "amount": body.amount,
        "description": f"Thu nợ đơn {o.get('order_code', body.order_id)} - {o.get('customer_name', '')}",
        "date": now.astimezone(vn_tz).isoformat(),
        "source": "debt_collect",
        "ref_id": payment["payment_id"],
        "created_at": now,
        "created_by": user.get("name"),
        "created_by_id": user["user_id"],
    }
    await db.finance_transactions.insert_one(finance_txn)
    # ─────────────────────────────────────────────────────────────────────────

    return {"order": await db.orders.find_one({"order_id": body.order_id}, {"_id": 0}), "payment": payment}


@api.post("/debts/pay-supplier")
async def debts_pay_supplier(body: DebtPaymentIn, user: dict = Depends(require_permission("debts.pay"))):
    if not body.supplier_id:
        raise HTTPException(status_code=400, detail="supplier_id is required")
    supplier = await db.suppliers.find_one({"supplier_id": body.supplier_id})
    if not supplier:
        raise HTTPException(status_code=404, detail="Không tìm thấy NCC")
    now = now_vn()
    payment = {
        "payment_id": f"pmt_{uuid.uuid4().hex[:10]}",
        "direction": "out",
        "supplier_id": body.supplier_id, "supplier_name": supplier.get("name"),
        "amount": body.amount, "method": body.method, "notes": body.notes,
        "created_at": now, "created_by": user.get("name"), "created_by_id": user["user_id"],
    }
    await db.debt_payments.insert_one(payment)
    # decrement supplier debt counter
    await db.suppliers.update_one({"supplier_id": body.supplier_id}, {"$inc": {"total_debt": -body.amount}})
    payment.pop("_id", None)
    return payment


@api.get("/debts/payments")
async def list_debt_payments(
    direction: Optional[str] = None,
    customer_id: Optional[str] = None,
    supplier_id: Optional[str] = None,
    user: dict = Depends(require_permission("debts.view")),
):
    q = {}
    if direction:
        q["direction"] = direction
    else:
        # Exclude legacy supplier_debt entries (from old stock-in auto-log) — only real payments now
        q["direction"] = {"$ne": "supplier_debt"}
    if customer_id:
        q["customer_id"] = customer_id
    if supplier_id:
        q["supplier_id"] = supplier_id
    items = await db.debt_payments.find(q, {"_id": 0}).sort("created_at", -1).limit(500).to_list(None)
    return items


# ===========================================================================
# Users
# ===========================================================================
@api.post("/users")
async def admin_create_user(body: RegisterIn, user: dict = Depends(require_permission("users.create"))):
    """Admin creates a new user — does NOT set auth cookies (avoids overwriting admin session)."""
    email = body.email.lower().strip()
    if await db.users.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="Email đã tồn tại")
    user_id = f"user_{uuid.uuid4().hex[:12]}"
    doc = {
        "user_id": user_id, "email": email, "name": body.name, "role": body.role,
        "phone": body.phone or "",
        "password_hash": hash_password(body.password),
        "auth_provider": "local",
        "is_active": True,
        "permissions": body.permissions if body.permissions is not None else None,
        "created_at": now_vn(),
    }
    await db.users.insert_one(doc)
    doc.pop("password_hash", None)
    doc.pop("_id", None)
    return doc


@api.get("/users")
async def list_users(user: dict = Depends(require_permission("users.view"))):
    items = await db.users.find({}, {"_id": 0, "password_hash": 0}).sort("created_at", -1).limit(500).to_list(None)
    return items


@api.put("/users/{user_id}")
async def update_user(user_id: str, body: UserUpdateIn, user: dict = Depends(require_permission("users.edit"))):
    upd = {k: v for k, v in body.model_dump().items() if v is not None and k != "password"}
    if body.password:
        upd["password_hash"] = hash_password(body.password)
    if upd:
        res = await db.users.update_one({"user_id": user_id}, {"$set": upd})
        if not res.matched_count:
            raise HTTPException(status_code=404, detail="Không tìm thấy user")
    return await db.users.find_one({"user_id": user_id}, {"_id": 0, "password_hash": 0})


@api.delete("/users/{user_id}")
async def delete_user(user_id: str, request_user: dict = Depends(require_permission("users.delete"))):
    if request_user["user_id"] == user_id:
        raise HTTPException(status_code=400, detail="Không thể tự xóa chính mình")
    res = await db.users.delete_one({"user_id": user_id})
    if not res.deleted_count:
        raise HTTPException(status_code=404, detail="Không tìm thấy user")
    return {"ok": True}


@api.get("/users/shippers")
async def list_shippers(user: dict = Depends(get_current_user)):
    items = await db.users.find(
        {"role": {"$in": ["shipper", "coordinator", "staff"]}, "is_active": {"$ne": False}},
        {"_id": 0, "password_hash": 0},
    ).limit(50).to_list(None)
    return items


# ===========================================================================
# Reports
# ===========================================================================
@api.get("/reports/revenue")
async def reports_revenue(
    period: str = "daily",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    user: dict = Depends(require_permission("reports.view")),
):
    now = now_vn().replace(hour=0, minute=0, second=0, microsecond=0)
    buckets = []
    if date_from and date_to:
        start_d = datetime.fromisoformat(date_from).replace(tzinfo=VN_TZ)
        end_d = datetime.fromisoformat(date_to).replace(tzinfo=VN_TZ)
        days = min((end_d - start_d).days + 1, 90)
        for i in range(max(days, 1)):
            d = start_d + timedelta(days=i)
            buckets.append((d, d + timedelta(days=1), d.strftime("%d/%m")))
    elif period == "daily":
        for i in range(29, -1, -1):
            d = now - timedelta(days=i)
            buckets.append((d, d + timedelta(days=1), d.strftime("%d/%m")))
    elif period == "weekly":
        # Tính từ đầu tuần hiện tại (thứ 2) rồi trừ lui 11 tuần
        days_since_monday = now.weekday()  # 0=Thứ 2
        week_start = now - timedelta(days=days_since_monday)
        for i in range(11, -1, -1):
            start = week_start - timedelta(weeks=i)
            end = start + timedelta(days=7)
            week_num = start.isocalendar()[1]
            buckets.append((start, end, f"T{week_num}/{start.year}"))
    elif period == "yearly":
        current_year = now.year
        for i in range(4, -1, -1):
            year = current_year - i
            start = datetime(year, 1, 1, tzinfo=VN_TZ)
            end = datetime(year + 1, 1, 1, tzinfo=VN_TZ)
            buckets.append((start, end, str(year)))
    else:
        for i in range(11, -1, -1):
            year = now.year
            month = now.month - i
            while month <= 0:
                month += 12
                year -= 1
            start = datetime(year, month, 1, tzinfo=VN_TZ)
            if month == 12:
                end = datetime(year + 1, 1, 1, tzinfo=VN_TZ)
            else:
                end = datetime(year, month + 1, 1, tzinfo=VN_TZ)
            buckets.append((start, end, f"{month:02d}/{year}"))

    result = []
    for start, end, label in buckets:
        orders = await db.orders.find(
            {"created_at": {"$gte": start, "$lt": end}, "status": {"$ne": "cancelled"}},
            {"_id": 0, "total": 1},
        ).limit(2000).to_list(None)
        result.append({"label": label, "revenue": sum(o.get("total", 0) for o in orders), "orders": len(orders)})
    return {"period": period, "data": result}


@api.get("/reports/debt")
async def reports_debt(user: dict = Depends(require_permission("reports.view"))):
    pipeline = [
        {"$match": {"payment_method": "debt", "is_paid": {"$ne": True}, "status": {"$ne": "cancelled"}}},
        {"$group": {
            "_id": "$customer_name", "phone": {"$first": "$customer_phone"},
            "debt": {"$sum": "$remaining_amount"},
            "orders": {"$sum": 1}, "last_order": {"$max": "$created_at"},
        }},
        {"$sort": {"debt": -1}},
        {"$limit": 50},
    ]
    items = []
    async for r in db.orders.aggregate(pipeline):
        items.append({
            "customer_name": r["_id"], "phone": r.get("phone"),
            "debt": r["debt"], "orders": r["orders"], "last_order": r.get("last_order"),
        })
    return {"items": items, "total": sum(i["debt"] for i in items)}


@api.get("/reports/inventory")
async def reports_inventory(user: dict = Depends(require_permission("reports.view"))):
    products = await db.products.find({"is_active": True}, {"_id": 0}).limit(1000).to_list(None)
    materials = await db.materials.find({"is_active": True}, {"_id": 0}).limit(1000).to_list(None)
    total_value = sum(p.get("stock", 0) * p.get("cost", 0) for p in products)
    mat_value = sum(m.get("stock", 0) * m.get("cost", 0) for m in materials)
    low = [p for p in products if 0 < p.get("stock", 0) <= p.get("low_stock_threshold", 0)]
    out = [p for p in products if p.get("stock", 0) == 0]
    negative = [p for p in products if p.get("stock", 0) < 0]
    return {
        "total_products": len(products),
        "total_stock_value": total_value,
        "low_stock_count": len(low),
        "out_of_stock_count": len(out),
        "negative_stock_count": len(negative),
        "negative_stock_products": negative,
        "products": products,
        "materials": materials,
        "total_materials_value": mat_value,
    }


# ===========================================================================
# AI Chat (Claude + OpenAI)
# ===========================================================================
async def get_business_context() -> str:
    today = now_vn().replace(hour=0, minute=0, second=0, microsecond=0)
    orders_today = await db.orders.count_documents({"created_at": {"$gte": today}})
    pending = await db.orders.count_documents({"status": {"$in": ["new", "processing"]}})
    delivering = await db.orders.count_documents({"status": "delivering"})
    low_stock = await db.products.find(
        {"$expr": {"$lte": ["$stock", "$low_stock_threshold"]}, "is_active": True}, {"_id": 0},
    ).limit(20).to_list(None)
    total_customers = await db.customers.count_documents({})
    total_products = await db.products.count_documents({"is_active": True})
    today_orders_docs = await db.orders.find(
        {"created_at": {"$gte": today}}, {"_id": 0, "total": 1, "status": 1},
    ).limit(2000).to_list(None)
    today_revenue = sum(o.get("total", 0) for o in today_orders_docs if o.get("status") != "cancelled")

    low_stock_text = "\n".join(
        [f"  - {p['name']}: còn {p['stock']} {p.get('unit', 'cái')}" for p in low_stock[:10]]
    ) or "  (không có)"

    return f"""DỮ LIỆU NỘI BỘ TIỆM BÁNH BAO (thời gian thực):
- Doanh thu hôm nay: {today_revenue:,.0f} VND
- Đơn hôm nay: {orders_today}
- Đơn đang xử lý: {pending}
- Đơn đang giao: {delivering}
- Tổng sản phẩm: {total_products}
- Tổng khách hàng: {total_customers}
- Sản phẩm sắp hết hàng:
{low_stock_text}
"""


MODEL_MAP = {
    "claude": ("anthropic", "claude-sonnet-4-5-20250929"),
    "claude-4-5": ("anthropic", "claude-sonnet-4-5-20250929"),
    "claude-4-6": ("anthropic", "claude-sonnet-4-6"),
    "openai": ("openai", "gpt-5.2"),
    "openai-5.2": ("openai", "gpt-5.2"),
    "gpt-5.2": ("openai", "gpt-5.2"),
    "gpt-5.1": ("openai", "gpt-5.1"),
    "gpt-4o": ("openai", "gpt-4o"),
}


@api.post("/chat")
async def chat(body: ChatIn, user: dict = Depends(get_current_user)):
    if not EMERGENT_LLM_KEY:
        raise HTTPException(status_code=500, detail="LLM key chưa được cấu hình")

    requested = (body.model or "claude").lower()
    provider, model_id = MODEL_MAP.get(requested, MODEL_MAP["claude"])

    context = await get_business_context()
    system_message = f"""Bạn là Bao - trợ lý AI nội bộ cho tiệm bánh bao Việt Nam. Trả lời ngắn gọn, súc tích, thân thiện.
Sử dụng dữ liệu thực tế bên dưới khi cần. Trả lời tiếng Việt mặc định.

{context}

Lưu ý: Trả lời ngắn (2-5 câu). Nếu cần dữ liệu chi tiết, đề nghị user kiểm tra trong module tương ứng."""

    chat_obj = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=f"{body.session_id}::{requested}",
        system_message=system_message,
    ).with_model(provider, model_id)

    history = await db.chat_messages.find(
        {"user_id": user["user_id"], "session_id": body.session_id, "model_key": requested},
        {"_id": 0},
    ).sort("created_at", 1).limit(20).to_list(None)

    try:
        for msg in history:
            if msg.get("role") == "user":
                await chat_obj.send_message(UserMessage(text=msg["content"]))
        response = await chat_obj.send_message(UserMessage(text=body.message))
    except Exception as e:
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail=f"Lỗi AI: {str(e)[:200]}")

    now = now_vn()
    await db.chat_messages.insert_many([
        {"user_id": user["user_id"], "session_id": body.session_id, "model_key": requested,
         "role": "user", "content": body.message, "created_at": now},
        {"user_id": user["user_id"], "session_id": body.session_id, "model_key": requested,
         "role": "assistant", "content": response, "created_at": now},
    ])

    return {"reply": response, "session_id": body.session_id, "model": f"{provider}/{model_id}"}


@api.get("/chat/history")
async def chat_history(session_id: str = Query(...), model: Optional[str] = None, user: dict = Depends(get_current_user)):
    q = {"user_id": user["user_id"], "session_id": session_id}
    if model:
        q["model_key"] = model.lower()
    msgs = await db.chat_messages.find(q, {"_id": 0}).sort("created_at", 1).limit(100).to_list(None)
    return msgs


# ===========================================================================
# Seed
# ===========================================================================
async def seed_data():
    # Indexes
    await db.users.create_index("email", unique=True)
    await db.users.create_index("user_id", unique=True)
    await db.products.create_index("product_id", unique=True)
    await db.materials.create_index("material_id", unique=True)
    await db.customers.create_index("customer_id", unique=True)
    await db.customers.create_index([("name", 1), ("phone", 1)])
    await db.suppliers.create_index("supplier_id", unique=True)
    await db.orders.create_index("order_id", unique=True)
    await db.orders.create_index("created_at")
    await db.orders.create_index("customer_id")
    await db.orders.create_index("payment_method")
    await db.stock_movements.create_index("created_at")
    await db.debt_payments.create_index("created_at")
    await db.finance_categories.create_index("category_id", unique=True)
    await db.finance_transactions.create_index("transaction_id", unique=True)
    await db.finance_transactions.create_index("date")
    await db.finance_transactions.create_index("type")
    await db.user_sessions.create_index("session_token")
    await db.login_attempts.create_index("identifier")

    now = now_vn()
    # Migrate legacy cod -> debt
    await db.orders.update_many({"payment_method": "cod"}, {"$set": {"payment_method": "debt"}})

    # Migrate legacy order statuses to new vocab (preparing -> processing)
    await db.orders.update_many({"status": "preparing"}, {"$set": {"status": "processing"}})

    # Backfill remaining_amount
    legacy_debt = await db.orders.find({"payment_method": "debt", "remaining_amount": {"$exists": False}}, {"order_id": 1, "total": 1, "is_paid": 1}).to_list(None)
    for o in legacy_debt:
        await db.orders.update_one(
            {"order_id": o["order_id"]},
            {"$set": {"remaining_amount": 0 if o.get("is_paid") else o.get("total", 0), "paid_amount": o.get("total", 0) if o.get("is_paid") else 0}},
        )

    # Seed admin
    admin = await db.users.find_one({"email": ADMIN_EMAIL})
    if not admin:
        await db.users.insert_one({
            "user_id": f"user_{uuid.uuid4().hex[:12]}",
            "email": ADMIN_EMAIL, "name": "Chủ tiệm", "role": "admin",
            "password_hash": hash_password(ADMIN_PASSWORD),
            "auth_provider": "local", "is_active": True,
            "created_at": now,
        })
        logger.info(f"Seeded admin {ADMIN_EMAIL}")
    else:
        if not verify_password(ADMIN_PASSWORD, admin.get("password_hash", "")):
            await db.users.update_one({"email": ADMIN_EMAIL}, {"$set": {"password_hash": hash_password(ADMIN_PASSWORD)}})

    # Seed default settings if missing
    s = await db.settings.find_one({"_singleton": SETTINGS_ID})
    if not s:
        defaults = ShopSettingsIn().model_dump()
        defaults["_singleton"] = SETTINGS_ID
        defaults["updated_at"] = now
        await db.settings.insert_one(defaults)

    # Seed default finance categories
    existing_cats = await db.finance_categories.count_documents({})
    if existing_cats == 0:
        default_cats = [
            {"category_id": f"fcat_{uuid.uuid4().hex[:10]}", "name": "Thu công nợ", "type": "income", "color": "#2D4A22", "description": "Tự động từ thu nợ", "created_at": now, "created_by": "system"},
            {"category_id": f"fcat_{uuid.uuid4().hex[:10]}", "name": "Doanh thu bán hàng", "type": "income", "color": "#5B8A3C", "description": "", "created_at": now, "created_by": "system"},
            {"category_id": f"fcat_{uuid.uuid4().hex[:10]}", "name": "Tiền điện nước", "type": "expense", "color": "#E07B39", "description": "", "created_at": now, "created_by": "system"},
            {"category_id": f"fcat_{uuid.uuid4().hex[:10]}", "name": "Nguyên liệu", "type": "expense", "color": "#C0392B", "description": "", "created_at": now, "created_by": "system"},
            {"category_id": f"fcat_{uuid.uuid4().hex[:10]}", "name": "Lương nhân viên", "type": "expense", "color": "#2980B9", "description": "", "created_at": now, "created_by": "system"},
            {"category_id": f"fcat_{uuid.uuid4().hex[:10]}", "name": "Chi phí vận chuyển", "type": "expense", "color": "#8E44AD", "description": "", "created_at": now, "created_by": "system"},
            {"category_id": f"fcat_{uuid.uuid4().hex[:10]}", "name": "Thuê mặt bằng", "type": "expense", "color": "#16A085", "description": "", "created_at": now, "created_by": "system"},
        ]
        await db.finance_categories.insert_many(default_cats)
        logger.info("Seeded default finance categories")

    # No more demo seeding — user wants real data only.


@app.on_event("startup")
async def on_startup():
    await seed_data()
    logger.info("Banh Bao Admin API v2 ready")


@app.on_event("shutdown")
async def on_shutdown():
    mongo_client.close()


@api.get("/")
async def root():
    return {"service": "Bánh Bao Admin API", "status": "ok", "version": "2.0"}


# ===========================================================================
# Finance — Danh mục chi phí & Thu Chi (Profit tracking)
# ===========================================================================

class FinanceCategoryIn(BaseModel):
    name: str
    type: Literal["income", "expense"] = "expense"
    color: Optional[str] = "#6B7280"
    description: Optional[str] = None

class FinanceTransactionIn(BaseModel):
    type: Literal["income", "expense"]
    category_id: Optional[str] = None
    amount: float
    description: Optional[str] = None
    date: Optional[str] = None  # ISO date string, defaults to now
    source: Optional[str] = None  # "manual" | "debt_collect"
    ref_id: Optional[str] = None  # payment_id if synced from debt

@api.get("/finance/categories")
async def list_finance_categories(user: dict = Depends(require_permission("reports.view"))):
    items = await db.finance_categories.find({}, {"_id": 0}).sort("name", 1).to_list(None)
    return items

@api.post("/finance/categories")
async def create_finance_category(body: FinanceCategoryIn, user: dict = Depends(require_permission("settings.edit"))):
    now = now_vn()
    cat = {
        "category_id": f"fcat_{uuid.uuid4().hex[:10]}",
        "name": body.name,
        "type": body.type,
        "color": body.color or "#6B7280",
        "description": body.description,
        "created_at": now,
        "created_by": user.get("name"),
    }
    await db.finance_categories.insert_one(cat)
    cat.pop("_id", None)
    return cat

@api.put("/finance/categories/{category_id}")
async def update_finance_category(category_id: str, body: FinanceCategoryIn, user: dict = Depends(require_permission("settings.edit"))):
    upd = {k: v for k, v in body.model_dump().items() if v is not None}
    res = await db.finance_categories.update_one({"category_id": category_id}, {"$set": upd})
    if not res.matched_count:
        raise HTTPException(status_code=404, detail="Không tìm thấy danh mục")
    return {"ok": True}

@api.delete("/finance/categories/{category_id}")
async def delete_finance_category(category_id: str, user: dict = Depends(require_permission("settings.edit"))):
    await db.finance_categories.delete_one({"category_id": category_id})
    return {"ok": True}

@api.get("/finance/transactions")
async def list_finance_transactions(
    period: Optional[str] = "monthly",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    type: Optional[str] = None,
    user: dict = Depends(require_permission("reports.view")),
):
    now = now_vn()
    vn_tz = timezone(timedelta(hours=7))
    now_local = now.astimezone(vn_tz)

    if date_from and date_to:
        start = datetime.fromisoformat(date_from).replace(tzinfo=vn_tz)
        end = datetime.fromisoformat(date_to).replace(tzinfo=vn_tz) + timedelta(days=1)
    elif period == "daily":
        start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        end = now_local.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif period == "weekly":
        start = (now_local - timedelta(days=now_local.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        end = now_local.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif period == "monthly":
        start = now_local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = now_local.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif period == "yearly":
        start = now_local.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        end = now_local.replace(hour=23, minute=59, second=59, microsecond=999999)
    else:
        start = now_local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = now_local.replace(hour=23, minute=59, second=59, microsecond=999999)

    q: Dict[str, Any] = {"date": {"$gte": start.isoformat(), "$lte": end.isoformat()}}
    if type:
        q["type"] = type

    items = await db.finance_transactions.find(q, {"_id": 0}).sort("date", -1).limit(1000).to_list(None)

    total_income = sum(i["amount"] for i in items if i["type"] == "income")
    total_expense = sum(i["amount"] for i in items if i["type"] == "expense")

    # Category breakdown for pie chart
    cats: Dict[str, Any] = {}
    for i in items:
        cid = i.get("category_id") or "__none__"
        cname = i.get("category_name") or ("Thu khác" if i["type"] == "income" else "Chi khác")
        ccolor = i.get("category_color") or "#6B7280"
        key = f"{i['type']}_{cid}"
        if key not in cats:
            cats[key] = {"category_id": cid, "category_name": cname, "type": i["type"], "color": ccolor, "total": 0, "count": 0}
        cats[key]["total"] += i["amount"]
        cats[key]["count"] += 1

    return {
        "transactions": items,
        "total_income": total_income,
        "total_expense": total_expense,
        "profit": total_income - total_expense,
        "category_breakdown": list(cats.values()),
    }

@api.post("/finance/transactions")
async def create_finance_transaction(body: FinanceTransactionIn, user: dict = Depends(require_permission("reports.view"))):
    now = now_vn()
    vn_tz = timezone(timedelta(hours=7))
    cat = None
    if body.category_id:
        cat = await db.finance_categories.find_one({"category_id": body.category_id}, {"_id": 0})

    txn = {
        "transaction_id": f"txn_{uuid.uuid4().hex[:10]}",
        "type": body.type,
        "category_id": body.category_id,
        "category_name": cat["name"] if cat else ("Thu khác" if body.type == "income" else "Chi khác"),
        "category_color": cat["color"] if cat else "#6B7280",
        "amount": body.amount,
        "description": body.description,
        "date": body.date or now.astimezone(vn_tz).isoformat(),
        "source": body.source or "manual",
        "ref_id": body.ref_id,
        "created_at": now,
        "created_by": user.get("name"),
        "created_by_id": user["user_id"],
    }
    await db.finance_transactions.insert_one(txn)
    txn.pop("_id", None)
    return txn

@api.delete("/finance/transactions/{transaction_id}")
async def delete_finance_transaction(transaction_id: str, user: dict = Depends(require_permission("settings.edit"))):
    await db.finance_transactions.delete_one({"transaction_id": transaction_id})
    return {"ok": True}


app.include_router(api)