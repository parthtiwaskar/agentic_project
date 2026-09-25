"""
agent_tools.py
Clean tool layer for the agentic AI hotel booking assistant.
Each tool validates inputs, returns structured JSON, and handles errors.
"""

import random
import string
from datetime import datetime, timedelta
from typing import Any, Optional


class HotelRegistry:
    """
    In-memory registry of hotels discovered during a session.
    Ensures the agent can only reference validated hotel IDs.
    """

    def __init__(self):
        self._hotels: dict[str, dict] = {}
        self._bookings: dict[str, dict] = {}

    def register_hotels(self, hotels: list[dict]) -> None:
        """Register hotels from a search result."""
        for h in hotels:
            hid = h.get("hotel_id")
            if hid:
                self._hotels[hid] = h

    def get_hotel(self, hotel_id: str) -> Optional[dict]:
        """Get a registered hotel by ID."""
        return self._hotels.get(hotel_id)

    def list_hotel_ids(self) -> list[str]:
        return list(self._hotels.keys())

    def register_booking(self, booking_id: str, booking: dict) -> None:
        self._bookings[booking_id] = booking

    def get_booking(self, booking_id: str) -> Optional[dict]:
        return self._bookings.get(booking_id)

    def clear(self) -> None:
        self._hotels.clear()
        self._bookings.clear()


def _err(message: str) -> dict:
    return {"success": False, "error": message}


def _ok(data: dict) -> dict:
    return {"success": True, **data}


# ── Tool: searchHotels ────────────────────────────────────────────────────

def tool_search_hotels(
    registry: HotelRegistry,
    search_fn,
    criteria: dict,
) -> dict[str, Any]:
    """
    Search for hotels matching criteria.
    criteria: {destination, checkIn, checkOut, guests, maxPrice, minPrice?, stars?}
    """
    destination = criteria.get("destination", "").strip()
    if not destination:
        return _err("Destination is required.")

    check_in = criteria.get("checkIn", criteria.get("check_in", ""))
    check_out = criteria.get("checkOut", criteria.get("check_out", ""))
    if not check_in or not check_out:
        return _err("Check-in and check-out dates are required.")

    guests = int(criteria.get("guests", criteria.get("adults", 2)))
    max_price = int(criteria.get("maxPrice", criteria.get("max_price", 500)))
    min_price = int(criteria.get("minPrice", criteria.get("min_price", 0)))
    stars = criteria.get("stars")
    if stars is not None and str(stars) not in ("Any", "None", "null", ""):
        stars = int(stars)
    else:
        stars = None

    if min_price > max_price:
        min_price, max_price = max_price, min_price

    hotels = search_fn(
        destination=destination,
        check_in=check_in,
        check_out=check_out,
        min_price=min_price,
        max_price=max_price,
        adults=guests,
        stars=stars,
    )

    if not hotels:
        return _ok({"hotels": [], "count": 0, "message": f"No hotels found in {destination} matching your criteria."})

    registry.register_hotels(hotels)

    # Return a summary for the agent (not the full data dump)
    hotel_summaries = []
    for h in hotels:
        hotel_summaries.append({
            "hotel_id": h.get("hotel_id"),
            "name": h.get("name"),
            "stars": h.get("stars"),
            "price_per_night": h.get("price"),
            "rating": h.get("rating"),
            "address": h.get("address"),
            "area": h.get("area", ""),
            "amenities": h.get("amenities", []),
            "cancellation_type": h.get("cancellation", {}).get("type", "unknown"),
        })

    return _ok({
        "hotels": hotel_summaries,
        "count": len(hotel_summaries),
        "destination": destination,
    })


# ── Tool: checkAvailability ───────────────────────────────────────────────

def tool_check_availability(
    registry: HotelRegistry,
    hotel_id: str,
    check_in: str,
    check_out: str,
    guests: int,
) -> dict[str, Any]:
    """Check availability for a specific hotel."""
    hotel = registry.get_hotel(hotel_id)
    if not hotel:
        return _err(f"Hotel ID '{hotel_id}' not found. Please search for hotels first.")

    # Simulated availability (90% chance available)
    available = random.random() < 0.90

    if not available:
        return _ok({
            "hotel_id": hotel_id,
            "hotel_name": hotel["name"],
            "available": False,
            "message": f"{hotel['name']} is not available for the selected dates.",
        })

    # Check guest capacity across rooms
    suitable_rooms = []
    for room in hotel.get("rooms", []):
        if room.get("max_guests", 2) >= guests:
            suitable_rooms.append(room["type"])

    if not suitable_rooms:
        return _ok({
            "hotel_id": hotel_id,
            "hotel_name": hotel["name"],
            "available": False,
            "message": f"No rooms at {hotel['name']} can accommodate {guests} guests.",
        })

    return _ok({
        "hotel_id": hotel_id,
        "hotel_name": hotel["name"],
        "available": True,
        "available_room_types": suitable_rooms,
        "check_in": check_in,
        "check_out": check_out,
        "guests": guests,
    })


# ── Tool: getHotelDetails ────────────────────────────────────────────────

def tool_get_hotel_details(
    registry: HotelRegistry,
    hotel_id: str,
) -> dict[str, Any]:
    """Get full details for a specific hotel."""
    hotel = registry.get_hotel(hotel_id)
    if not hotel:
        return _err(f"Hotel ID '{hotel_id}' not found. Please search for hotels first.")

    return _ok({
        "hotel_id": hotel_id,
        "name": hotel["name"],
        "stars": hotel["stars"],
        "rating": hotel["rating"],
        "address": hotel["address"],
        "destination": hotel.get("destination", ""),
        "area": hotel.get("area", ""),
        "landmark": hotel.get("landmark", ""),
        "distance_km": hotel.get("distance_km", 0),
        "amenities": hotel.get("amenities", []),
        "rooms": hotel.get("rooms", []),
        "cancellation": hotel.get("cancellation", {}),
        "base_price_per_night": hotel["price"],
    })


# ── Tool: calculateBookingCost ───────────────────────────────────────────

def tool_calculate_booking_cost(
    registry: HotelRegistry,
    hotel_id: str,
    room_type: str,
    check_in: str,
    check_out: str,
) -> dict[str, Any]:
    """Calculate total booking cost including taxes."""
    hotel = registry.get_hotel(hotel_id)
    if not hotel:
        return _err(f"Hotel ID '{hotel_id}' not found.")

    # Find room
    room_match = None
    for r in hotel.get("rooms", []):
        if r["type"].lower() == room_type.lower():
            room_match = r
            break

    if not room_match:
        available = [r["type"] for r in hotel.get("rooms", [])]
        return _err(f"Room type '{room_type}' not available. Available: {', '.join(available)}")

    # Calculate nights
    try:
        ci = datetime.strptime(check_in, "%Y-%m-%d")
        co = datetime.strptime(check_out, "%Y-%m-%d")
        nights = (co - ci).days
        if nights <= 0:
            return _err("Check-out must be after check-in.")
    except ValueError:
        return _err("Invalid date format. Use YYYY-MM-DD.")

    price_per_night = room_match["price_per_night"]
    subtotal = price_per_night * nights
    tax_rate = 0.12
    taxes = round(subtotal * tax_rate)
    total = subtotal + taxes

    return _ok({
        "hotel_id": hotel_id,
        "hotel_name": hotel["name"],
        "room_type": room_match["type"],
        "check_in": check_in,
        "check_out": check_out,
        "nights": nights,
        "price_per_night": price_per_night,
        "subtotal": subtotal,
        "tax_rate": f"{int(tax_rate * 100)}%",
        "taxes": taxes,
        "total": total,
        "currency": "USD",
    })


# ── Tool: checkCancellationPolicy ────────────────────────────────────────

def tool_check_cancellation_policy(
    registry: HotelRegistry,
    hotel_id: str,
    room_type: str = "",
) -> dict[str, Any]:
    """Check cancellation policy for a hotel."""
    hotel = registry.get_hotel(hotel_id)
    if not hotel:
        return _err(f"Hotel ID '{hotel_id}' not found.")

    policy = hotel.get("cancellation", {})
    return _ok({
        "hotel_id": hotel_id,
        "hotel_name": hotel["name"],
        "cancellation_type": policy.get("type", "unknown"),
        "deadline_days": policy.get("deadline_days", 0),
        "description": policy.get("description", "No policy information available."),
    })


# ── Tool: getHotelLocation ──────────────────────────────────────────────

def tool_get_hotel_location(
    registry: HotelRegistry,
    hotel_id: str,
) -> dict[str, Any]:
    """Get location details for a hotel."""
    hotel = registry.get_hotel(hotel_id)
    if not hotel:
        return _err(f"Hotel ID '{hotel_id}' not found.")

    return _ok({
        "hotel_id": hotel_id,
        "hotel_name": hotel["name"],
        "address": hotel.get("address", ""),
        "area": hotel.get("area", ""),
        "landmark": hotel.get("landmark", ""),
        "distance_km": hotel.get("distance_km", 0),
        "destination": hotel.get("destination", ""),
    })


# ── Tool: createBooking ─────────────────────────────────────────────────

def tool_create_booking(
    registry: HotelRegistry,
    booking_details: dict,
) -> dict[str, Any]:
    """
    Create a booking after user confirmation.
    booking_details: {hotel_id, room_type, check_in, check_out, guests, total_cost}
    """
    hotel_id = booking_details.get("hotel_id", "")
    hotel = registry.get_hotel(hotel_id)
    if not hotel:
        return _err(f"Hotel ID '{hotel_id}' not found. Cannot create booking.")

    room_type = booking_details.get("room_type", "")
    check_in = booking_details.get("check_in", "")
    check_out = booking_details.get("check_out", "")
    guests = booking_details.get("guests", 0)

    if not all([room_type, check_in, check_out, guests]):
        return _err("Missing required booking details: room_type, check_in, check_out, guests.")

    # Verify room exists
    room_match = None
    for r in hotel.get("rooms", []):
        if r["type"].lower() == room_type.lower():
            room_match = r
            break
    if not room_match:
        return _err(f"Room type '{room_type}' not found at {hotel['name']}.")

    # Final availability check (95% success on second check)
    if random.random() < 0.05:
        return _ok({
            "booked": False,
            "hotel_id": hotel_id,
            "hotel_name": hotel["name"],
            "reason": "unavailable",
            "message": f"Unfortunately, {hotel['name']} just became unavailable. Please try an alternative.",
        })

    # Generate booking ID
    booking_id = "HTL-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6))

    # Calculate cost
    try:
        ci = datetime.strptime(check_in, "%Y-%m-%d")
        co = datetime.strptime(check_out, "%Y-%m-%d")
        nights = (co - ci).days
    except ValueError:
        nights = 1

    price_per_night = room_match["price_per_night"]
    subtotal = price_per_night * nights
    taxes = round(subtotal * 0.12)
    total = subtotal + taxes

    booking = {
        "booking_id": booking_id,
        "hotel_id": hotel_id,
        "hotel_name": hotel["name"],
        "room_type": room_match["type"],
        "check_in": check_in,
        "check_out": check_out,
        "nights": nights,
        "guests": guests,
        "price_per_night": price_per_night,
        "subtotal": subtotal,
        "taxes": taxes,
        "total": total,
        "currency": "USD",
        "status": "confirmed",
        "cancellation": hotel.get("cancellation", {}),
        "created_at": datetime.now().isoformat(),
    }

    registry.register_booking(booking_id, booking)

    return _ok({
        "booked": True,
        "booking": booking,
    })


# ── Tool: getBookingStatus ───────────────────────────────────────────────

def tool_get_booking_status(
    registry: HotelRegistry,
    booking_id: str,
) -> dict[str, Any]:
    """Get the status of a booking."""
    booking = registry.get_booking(booking_id)
    if not booking:
        return _err(f"Booking ID '{booking_id}' not found.")

    return _ok({
        "booking": booking,
    })


# ── Tool Definitions (for the agent system prompt) ───────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "searchHotels",
        "description": "Search for hotels matching the user's criteria. Returns a list of matching hotels.",
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {"type": "string", "description": "City or location name"},
                "checkIn": {"type": "string", "description": "Check-in date (YYYY-MM-DD)"},
                "checkOut": {"type": "string", "description": "Check-out date (YYYY-MM-DD)"},
                "guests": {"type": "integer", "description": "Number of guests"},
                "maxPrice": {"type": "integer", "description": "Maximum price per night in USD"},
                "minPrice": {"type": "integer", "description": "Minimum price per night in USD"},
                "stars": {"type": "integer", "description": "Star rating filter (1-5), null for any"},
            },
            "required": ["destination", "checkIn", "checkOut", "guests", "maxPrice"],
        },
    },
    {
        "name": "checkAvailability",
        "description": "Check if a specific hotel is available for the given dates and guests.",
        "parameters": {
            "type": "object",
            "properties": {
                "hotelId": {"type": "string", "description": "The hotel's unique ID"},
                "checkIn": {"type": "string", "description": "Check-in date (YYYY-MM-DD)"},
                "checkOut": {"type": "string", "description": "Check-out date (YYYY-MM-DD)"},
                "guests": {"type": "integer", "description": "Number of guests"},
            },
            "required": ["hotelId", "checkIn", "checkOut", "guests"],
        },
    },
    {
        "name": "getHotelDetails",
        "description": "Get full details for a hotel including rooms, amenities, cancellation policy, and location.",
        "parameters": {
            "type": "object",
            "properties": {
                "hotelId": {"type": "string", "description": "The hotel's unique ID"},
            },
            "required": ["hotelId"],
        },
    },
    {
        "name": "calculateBookingCost",
        "description": "Calculate the total booking cost including taxes for a specific hotel and room.",
        "parameters": {
            "type": "object",
            "properties": {
                "hotelId": {"type": "string", "description": "The hotel's unique ID"},
                "roomType": {"type": "string", "description": "Room type name"},
                "checkIn": {"type": "string", "description": "Check-in date (YYYY-MM-DD)"},
                "checkOut": {"type": "string", "description": "Check-out date (YYYY-MM-DD)"},
            },
            "required": ["hotelId", "roomType", "checkIn", "checkOut"],
        },
    },
    {
        "name": "checkCancellationPolicy",
        "description": "Check the cancellation policy for a hotel.",
        "parameters": {
            "type": "object",
            "properties": {
                "hotelId": {"type": "string", "description": "The hotel's unique ID"},
                "roomType": {"type": "string", "description": "Room type (optional)"},
            },
            "required": ["hotelId"],
        },
    },
    {
        "name": "getHotelLocation",
        "description": "Get location details for a hotel (area, landmark, distance).",
        "parameters": {
            "type": "object",
            "properties": {
                "hotelId": {"type": "string", "description": "The hotel's unique ID"},
            },
            "required": ["hotelId"],
        },
    },
    {
        "name": "createBooking",
        "description": "Create a confirmed booking. Only call AFTER user explicitly confirms. Requires hotel_id, room_type, check_in, check_out, guests.",
        "parameters": {
            "type": "object",
            "properties": {
                "hotel_id": {"type": "string", "description": "The hotel's unique ID"},
                "room_type": {"type": "string", "description": "Room type name"},
                "check_in": {"type": "string", "description": "Check-in date (YYYY-MM-DD)"},
                "check_out": {"type": "string", "description": "Check-out date (YYYY-MM-DD)"},
                "guests": {"type": "integer", "description": "Number of guests"},
            },
            "required": ["hotel_id", "room_type", "check_in", "check_out", "guests"],
        },
    },
    {
        "name": "getBookingStatus",
        "description": "Get the status and details of an existing booking.",
        "parameters": {
            "type": "object",
            "properties": {
                "bookingId": {"type": "string", "description": "The booking ID (HTL-XXXXXX)"},
            },
            "required": ["bookingId"],
        },
    },
]

GROQ_TOOLS = [{"type": "function", "function": t} for t in TOOL_DEFINITIONS]

