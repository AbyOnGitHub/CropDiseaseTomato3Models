"""
Comprehensive agronomic knowledge base for the 6 tomato leaf conditions.
Includes pathogen classification, visual symptoms, favorable microclimates,
and actionable multi-tiered remediation measures for farmers.
"""

TOMATO_DISEASES = {
    "Tomato Early Blight": {
        "id": "early_blight",
        "canonical_name": "Tomato Early Blight",
        "scientific_name": "Alternaria solani",
        "pathogen_type": "Fungal Pathogen",
        "severity": "High",
        "severity_color": "#f97316", # Orange
        "is_healthy": False,
        "summary": "Common fungal disease causing dark brown spots with concentric target-board rings on older foliage, leading to premature defoliation and yield loss.",
        "symptoms": [
            "Dark brown to black spots with concentric rings ('bullseye' pattern) on older lower leaves.",
            "Yellow chlorotic halos surrounding leaf lesions.",
            "Leaves turn completely yellow, wither, and drop, progressing upward from the bottom of the plant.",
            "Stem lesions can cause collar rot in young seedlings and stem cankers in mature plants.",
            "Dark, leathery sunken lesions near the stem end of fruits."
        ],
        "favorable_conditions": "Warm temperatures (24°C - 29°C / 75°F - 85°F) with high humidity, prolonged leaf wetness, and rainfall splash.",
        "cultural_remediation": [
            "Prune off and safely destroy infected lower leaves immediately; do not compost infected foliage.",
            "Use drip irrigation or soaker hoses exclusively at soil level; avoid overhead sprinkler watering.",
            "Apply a 2-3 inch layer of clean straw, wood mulch, or plastic mulch around the plant base to prevent rain splashback of soil-borne spores.",
            "Stake, cage, or trellis plants to elevate foliage off the ground and maximize air circulation."
        ],
        "organic_treatments": [
            "Copper Octanoate (Copper soap) or Copper Hydroxide spray applied at 7-10 day intervals.",
            "Bio-fungicides containing Bacillus subtilis (e.g., Serenade) or Trichoderma harzianum as protective soil/foliar drenches.",
            "Potassium bicarbonate foliar spray (3g/L water) to elevate leaf surface pH and suppress spore germination."
        ],
        "chemical_treatments": [
            "Protective fungicides: Chlorothalonil or Mancozeb applied preventively before rain events.",
            "Curative/systemic fungicides: Azoxystrobin, Difenoconazole, or Boscalid if infection has already established.",
            "Strictly observe Pre-Harvest Interval (PHI) of 1-7 days depending on the selected commercial label."
        ],
        "preventive_measures": [
            "Implement a strict 2-3 year crop rotation excluding Solanaceae crops (tomatoes, potatoes, peppers, eggplants).",
            "Plant certified disease-free, high-vigor seedlings and resistant cultivars (e.g., Mountain Supreme, Defiant).",
            "Thoroughly plow down or remove all crop debris immediately following final harvest."
        ]
    },

    "Tomato Healthy": {
        "id": "healthy",
        "canonical_name": "Tomato Healthy",
        "scientific_name": "Solanum lycopersicum (Healthy Foliage)",
        "pathogen_type": "None (Optimum Physiological State)",
        "severity": "Optimal",
        "severity_color": "#22c55e", # Green
        "is_healthy": True,
        "summary": "Foliage exhibits vigorous, uniform green pigmentation with no detectable fungal lesions, bacterial speckling, or viral distortion.",
        "symptoms": [
            "Deep, uniform green leaf blades with smooth, intact margins.",
            "Strong turgidity and active apical vegetative growth.",
            "No chlorosis, necrosis, curling, wilting, or fungal sporulation.",
            "Healthy floral clusters with high fruit set potential."
        ],
        "favorable_conditions": "Well-balanced sunlight (6-8 hours daily), 20°C - 28°C daytime temperatures, and consistent soil moisture (60-70% field capacity).",
        "cultural_remediation": [
            "Maintain consistent drip irrigation schedule; avoid water stress fluctuations that induce blossom end rot.",
            "Prune non-fruiting lower suckers up to the first flower cluster to promote good air ventilation through the canopy.",
            "Inspect underside of leaves twice weekly to detect early arrival of silverleaf whiteflies, aphids, or hornworms.",
            "Ensure regular staking/trellising adjustments as vines expand."
        ],
        "organic_treatments": [
            "Apply prophylactic foliar seaweed extract or fish emulsion every 2 weeks for enhanced plant immunity.",
            "Prophylactic foliar spray of cold-pressed neem oil (0.5%) once a month to repel exploratory insect vectors."
        ],
        "chemical_treatments": [
            "No chemical fungicides or viricides required.",
            "Maintain balanced N-P-K fertigation (e.g., 10-10-20 during fruiting) along with soluble calcium nitrate (CaNO3) and magnesium sulfate (Epsom salts)."
        ],
        "preventive_measures": [
            "Scout crop weekly for early pest ingress.",
            "Keep field borders free of solanaceous weeds (nightshade, jimsonweed) that host viruses.",
            "Disinfect pruning shears with 70% isopropyl alcohol between rows."
        ]
    },

    "Tomato Late Blight": {
        "id": "late_blight",
        "canonical_name": "Tomato Late Blight",
        "scientific_name": "Phytophthora infestans",
        "pathogen_type": "Oomycete Pathogen (High Risk / Devastating)",
        "severity": "Critical",
        "severity_color": "#ef4444", # Red
        "is_healthy": False,
        "summary": "Extremely aggressive water mold causing rapid blighting, greasy olive-brown leaf lesions with white fuzzy sporulation beneath, and firm rot in green fruit.",
        "symptoms": [
            "Irregular, water-soaked pale-to-dark olive/brown lesions that enlarge rapidly.",
            "Fine, delicate white fuzzy downy mildew on the underside of leaves under damp morning humidity.",
            "Stems develop dark brown to black greasy water-soaked girdling lesions.",
            "Fruits develop large, firm, bumpy golden-brown to dark leathery blotches.",
            "Entire plants can collapse and turn brown within 48-72 hours under wet weather."
        ],
        "favorable_conditions": "Cool to moderate temperatures (15°C - 22°C / 60°F - 72°F) accompanied by high relative humidity (>90%), morning fogs, and continuous rain.",
        "cultural_remediation": [
            "CRITICAL EMERGENCY: Late blight spreads rapidly on the wind. Immediately remove and seal heavily infected plants inside plastic bags; bury or burn them. Never leave on the ground.",
            "Shut off all overhead irrigation immediately; keep foliage bone dry.",
            "Widen row spacing and prune excess vegetative foliage to accelerate drying time after rain.",
            "Notify neighboring growers if late blight is confirmed in your locality."
        ],
        "organic_treatments": [
            "Fixed copper compounds (Copper Hydroxide, Bordeaux mixture) provide surface protective coverage but must be applied BEFORE spore arrival.",
            "Bio-fungicides have limited efficacy once late blight is actively sporulating; use organic copper strictly as a preventive barrier."
        ],
        "chemical_treatments": [
            "Preventive barrier: Mancozeb or Chlorothalonil sprayed before forecast rain periods.",
            "Curative systemic oomycete fungicides: Dimethomorph, Metalaxyl-M / Mefenoxam (Ridomil Gold), Cymoxanil (Curzate), or Fluopicolide.",
            "Rotate chemical modes of action (FRAC groups 4, 11, 27, 43) to prevent resistance development."
        ],
        "preventive_measures": [
            "Plant resistant tomato hybrids (e.g., Defiant Ph-R, Iron Lady, Mountain Merit).",
            "Eliminate volunteer tomato and potato cull piles within 500 meters of fields.",
            "Check local university extension disease forecasting alerts (e.g., Blitecast)."
        ]
    },

    "Tomato Leaf Mold": {
        "id": "leaf_mold",
        "canonical_name": "Tomato Leaf Mold",
        "scientific_name": "Passalora fulva (syn. Cladosporium fulvum)",
        "pathogen_type": "Fungal Pathogen (High-Humidity Specialist)",
        "severity": "Medium",
        "severity_color": "#eab308", # Yellow
        "is_healthy": False,
        "summary": "Foliar fungus thriving in greenhouse and tunnel environments, forming pale yellow spots on leaf tops and velvety olive-brown spore mats underneath.",
        "symptoms": [
            "Diffuse pale green to chlorotic yellow blotches on the upper leaf surface.",
            "Corresponding dense, velvety olive-green to grayish-brown fungal spore carpets on the leaf undersides.",
            "Older leaves become necrotic, curl upwards, and drop prematurely.",
            "Blossom blighting and rare fruit calyx rot in heavy infestations."
        ],
        "favorable_conditions": "Warm and humid microclimates with relative humidity exceeding 85% and temperatures between 21°C - 24°C (common in greenhouses, polytunnels, and dense canopies).",
        "cultural_remediation": [
            "Vigorously increase ventilation: open greenhouse ridge vents, raise side curtains, and deploy horizontal airflow (HAF) fans.",
            "Reduce relative humidity below 85% by heating greenhouse air at dawn and sunset.",
            "Prune bottom leaves up to 18 inches off the ground to establish unrestricted air circulation under the crop canopy.",
            "Water exclusively in early morning so moisture on surfaces dries before midday."
        ],
        "organic_treatments": [
            "Copper Octanoate or Copper Soap spray, ensuring targeted coverage directly on the leaf undersides.",
            "Bio-fungicide sprays containing Bacillus amyloliquefaciens strain D747 or Streptomyces lydicus.",
            "Hydrogen dioxide / peroxyacetic acid foliar sanitizing spray."
        ],
        "chemical_treatments": [
            "Protective fungicides: Chlorothalonil or Mancozeb applied upon initial spot detection.",
            "Targeted translaminar fungicides: Difenoconazole, Cyprodinil + Fludioxonil (Switch), or Boscalid.",
            "Ensure spray nozzles are angled upward to wet the lower leaf surfaces."
        ],
        "preventive_measures": [
            "Select tomato varieties possessing specific Cf resistance genes (Cf-2, Cf-4, Cf-9).",
            "Thoroughly clean and sanitize greenhouse interior surfaces and support wires with quaternary ammonium or chlorine dioxide between crop cycles.",
            "Avoid planting at excessive densities inside closed high tunnels."
        ]
    },

    "Tomato Septoria Spot": {
        "id": "septoria_spot",
        "canonical_name": "Tomato Septoria Spot",
        "scientific_name": "Septoria lycopersici",
        "pathogen_type": "Fungal Leaf Spot",
        "severity": "High",
        "severity_color": "#f97316", # Orange
        "is_healthy": False,
        "summary": "Destructive fungal leaf spot causing numerous small circular lesions with grayish centers and tiny black fruiting bodies, causing rapid bottom-up defoliation.",
        "symptoms": [
            "Numerous small, circular lesions (1.5mm to 3mm wide) on lower leaves.",
            "Spots have dark brown borders with light tan to ash-gray centers.",
            "Tiny black pepper-like specks (pycnidia / spore-bearing structures) visible in the center of mature spots under magnifying glass.",
            "Affected leaves rapidly turn yellow, curl, wither, and drop.",
            "Defoliation exposes ripening fruit to severe sunscald."
        ],
        "favorable_conditions": "Prolonged wet weather with temperatures of 20°C - 25°C (68°F - 77°F). Spores splash from soil and debris via rain droplets or overhead sprinklers.",
        "cultural_remediation": [
            "Promptly clip off lower infected leaves with clean shears as soon as the first 1-2 spots appear.",
            "Rake up and destroy fallen leaf litter beneath plants regularly.",
            "Apply clean straw or plastic mulch to create a physical barrier preventing rain from splashing soil-borne spores onto leaves.",
            "Never work in or harvest tomato fields while plants are wet with dew or rain."
        ],
        "organic_treatments": [
            "Copper Sulfate / Copper Hydroxide spray applied on a 7-day schedule.",
            "Bacillus subtilis or Bacillus amyloliquefaciens biological fungicides applied weekly.",
            "Potassium bicarbonate foliar spray for organic suppression."
        ],
        "chemical_treatments": [
            "Fungicides with broad-spectrum activity: Chlorothalonil, Mancozeb, or Ziram.",
            "Systemic strobilurin / triazole fungicides: Azoxystrobin (Quadris), Pyraclostrobin (Cabrio), or Difenoconazole (Inspire Super).",
            "Begin treatments at first sign of disease and repeat according to label intervals during rainy periods."
        ],
        "preventive_measures": [
            "Practice a 3-year crop rotation away from solanaceous plants.",
            "Control perennial solanaceous weeds such as horse nettle and black nightshade.",
            "Deep-till or remove tomato crop residue at the end of the season to bury overwintering pycnidia."
        ]
    },

    "Tomato Yellow Leaf Curl Virus": {
        "id": "yellow_leaf_curl_virus",
        "canonical_name": "Tomato Yellow Leaf Curl Virus",
        "scientific_name": "Tomato Yellow Leaf Curl Virus (TYLCV / Begomovirus)",
        "pathogen_type": "Viral Pathogen (Vector-Borne by Whitefly)",
        "severity": "Critical",
        "severity_color": "#dc2626", # Deep Red
        "is_healthy": False,
        "summary": "Devastating viral disease transmitted by silverleaf whiteflies, causing severe stunting, erect bushy growth, cupped yellow leaves, and complete cessation of fruit set.",
        "symptoms": [
            "Severe upward rolling and curling of leaf margins, giving leaves a cupped or 'spoon-like' appearance.",
            "Pronounced interveinal chlorosis (bright yellowing) along leaf edges and margins.",
            "Marked reduction in leaf size and puckered, leathery leaf texture.",
            "Severe plant stunting with shortened internodes, creating a bushy, erect growth habit.",
            "Abscission (dropping) of flowers, leading to total failure of fruit set if infected prior to flowering."
        ],
        "favorable_conditions": "Thrives in warm, dry weather that favors high populations of its sole insect vector: the Silverleaf Whitefly (Bemisia tabaci).",
        "cultural_remediation": [
            "ROUGING IS ESSENTIAL: Infected plants CANNOT be cured of the virus. Immediately pull up infected plants, seal in airtight bags, and destroy them so they do not infect adjacent plants.",
            "Hang yellow sticky cards (1 per 10-20 sq meters) at canopy height to continuously monitor and trap adult whiteflies.",
            "Install 50-mesh fine insect exclusion screens over all greenhouse vents and double-door airlocks.",
            "Eliminate weed hosts (mallow, sowthistle, nightshade) in and around the perimeter of production fields."
        ],
        "organic_treatments": [
            "Spray insecticidal soaps or horticultural oils (e.g., JMS Stylet-Oil) at 5-7 day intervals to smother whitefly nymphs and eggs.",
            "Apply high-grade cold-pressed neem oil (Azadirachtin) or entomopathogenic fungal biopesticides (Beauveria bassiana, e.g., Botanigard).",
            "Introduce biological control predators in greenhouses: parasitic wasps (Encarsia formosa, Eretmocerus eremicus) or predatory mites (Amblyseius swirskii)."
        ],
        "chemical_treatments": [
            "Target the whitefly vector with registered systemic insecticides: Imidacloprid, Thiamethoxam, Dinotefuran (neonicotinoids) as drench or drip.",
            "Foliar sprays: Spiromesifen, Pyriproxyfen (insect growth regulators), Flonicamid, or Cyantraniliprole.",
            "Rotate insecticidal classes to prevent whitefly pesticide resistance."
        ],
        "preventive_measures": [
            "Plant TYLCV-resistant or tolerant tomato hybrids possessing Ty-1, Ty-2, or Ty-3 resistance genes (e.g., Tygress, Charger, Invictus, SV7846TH).",
            "Use reflective silver or metallic plastic mulch to disorient incoming whiteflies and reduce landing rates.",
            "Establish a mandatory 2-4 month crop-free host break between consecutive tomato plantings."
        ]
    }
}

# Unified label matching index
LABEL_MAP = {
    # Early blight aliases
    "tomato_early_blight": "Tomato Early Blight",
    "tomato early blight": "Tomato Early Blight",
    "early blight": "Tomato Early Blight",
    "early_blight": "Tomato Early Blight",
    "0": "Tomato Early Blight", # Note: will resolve per-model mapping below
    
    # Healthy aliases
    "tomato_healthy": "Tomato Healthy",
    "tomato healthy": "Tomato Healthy",
    "healthy": "Tomato Healthy",

    # Late blight aliases
    "tomato_late_blight": "Tomato Late Blight",
    "tomato late blight": "Tomato Late Blight",
    "late blight": "Tomato Late Blight",
    "late_blight": "Tomato Late Blight",

    # Leaf mold aliases
    "tomato_leaf_mold": "Tomato Leaf Mold",
    "tomato leaf mold": "Tomato Leaf Mold",
    "leaf mold": "Tomato Leaf Mold",
    "leaf_mold": "Tomato Leaf Mold",

    # Septoria spot aliases
    "tomato_septoria_spot": "Tomato Septoria Spot",
    "tomato septoria spot": "Tomato Septoria Spot",
    "septoria spot": "Tomato Septoria Spot",
    "septoria_spot": "Tomato Septoria Spot",

    # TYLCV aliases
    "tomato_yellow_leaf_curl_virus": "Tomato Yellow Leaf Curl Virus",
    "tomato yellow leaf curl virus": "Tomato Yellow Leaf Curl Virus",
    "yellow leaf curl virus": "Tomato Yellow Leaf Curl Virus",
    "yellow_leaf_curl_virus": "Tomato Yellow Leaf Curl Virus",
}

def get_disease_info(canonical_name: str) -> dict:
    """Retrieve full disease details and agronomic remediation measures."""
    return TOMATO_DISEASES.get(canonical_name, TOMATO_DISEASES["Tomato Healthy"])
