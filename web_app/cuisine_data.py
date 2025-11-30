"""Cuisine recommendations based on regional accents"""

ACCENT_TO_CUISINE = {
    "andhra_pradesh": {
        "region": "Andhra Pradesh",
        "native_language": "Telugu",
        "cuisines": [
            {"name": "Hyderabadi Biryani", "description": "Aromatic rice with marinated meat"},
            {"name": "Pulihora", "description": "Tangy tamarind rice"},
            {"name": "Gongura Chutney", "description": "Spicy sorrel leaves chutney"},
            {"name": "Pesarattu", "description": "Green gram dosa"},
            {"name": "Gutti Vankaya", "description": "Stuffed brinjal curry"}
        ]
    },
    "gujrat": {
        "region": "Gujarat",
        "native_language": "Gujarati",
        "cuisines": [
            {"name": "Dhokla", "description": "Steamed fermented gram flour cake"},
            {"name": "Thepla", "description": "Spiced flatbread with fenugreek"},
            {"name": "Khandvi", "description": "Rolled gram flour snack"},
            {"name": "Undhiyu", "description": "Mixed vegetable curry"},
            {"name": "Gujarati Kadhi", "description": "Sweet and tangy yogurt curry"}
        ]
    },
    "jharkhand": {
        "region": "Jharkhand",
        "native_language": "Hindi/Tribal languages",
        "cuisines": [
            {"name": "Litti Chokha", "description": "Roasted wheat balls with mashed vegetables"},
            {"name": "Dhuska", "description": "Deep-fried rice pancake"},
            {"name": "Chilka Roti", "description": "Rice flour flatbread"},
            {"name": "Rugra", "description": "Mushroom curry"},
            {"name": "Bamboo Shoot Curry", "description": "Traditional tribal dish"}
        ]
    },
    "karnataka": {
        "region": "Karnataka",
        "native_language": "Kannada",
        "cuisines": [
            {"name": "Mysore Masala Dosa", "description": "Crispy dosa with spicy red chutney"},
            {"name": "Bisi Bele Bath", "description": "Spicy lentil rice"},
            {"name": "Ragi Mudde", "description": "Finger millet balls"},
            {"name": "Mangalore Buns", "description": "Sweet banana puris"},
            {"name": "Neer Dosa", "description": "Thin rice crepes"}
        ]
    },
    "kerala": {
        "region": "Kerala",
        "native_language": "Malayalam",
        "cuisines": [
            {"name": "Appam with Stew", "description": "Fermented rice pancakes with coconut curry"},
            {"name": "Puttu", "description": "Steamed rice cakes"},
            {"name": "Avial", "description": "Mixed vegetable coconut curry"},
            {"name": "Karimeen Pollichathu", "description": "Pearl spot fish wrapped in banana leaf"},
            {"name": "Sadya", "description": "Traditional feast on banana leaf"}
        ]
    },
    "tamil": {
        "region": "Tamil Nadu",
        "native_language": "Tamil",
        "cuisines": [
            {"name": "Chettinad Chicken", "description": "Spicy chicken curry"},
            {"name": "Sambar", "description": "Lentil vegetable stew"},
            {"name": "Idli & Dosa", "description": "Fermented rice cakes and crepes"},
            {"name": "Pongal", "description": "Rice and lentil dish"},
            {"name": "Filter Coffee", "description": "Traditional South Indian coffee"}
        ]
    }
}

def get_cuisine_recommendations(predicted_accent):
    """Get cuisine recommendations based on detected accent"""
    accent_key = predicted_accent.lower().replace(" ", "_")
    
    if accent_key in ACCENT_TO_CUISINE:
        return ACCENT_TO_CUISINE[accent_key]
    else:
        return {
            "region": "Unknown",
            "native_language": "Unknown",
            "cuisines": [{"name": "General Indian Cuisine", "description": "Variety of Indian dishes"}]
        }
