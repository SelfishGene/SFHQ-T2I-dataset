#%%

import random
import numpy as np

#%% data

lighting_descriptions_dict = {
    'Natural Lighting': [
        "Illuminated by the soft glow of natural sunlight",
        "Bathed in the gentle embrace of daylight",
        "Highlighted by the subtle nuances of natural light",
    ],
    'Artificial Lighting': [
        "Enhanced by the steady gleam of artificial light sources",
        "Illuminated by the consistent radiance of man-made lighting",
        "Under the clear influence of artificial luminance",
        "Bathed in the consistent glow of artificial light sources",
        "Illuminated by artificial lights that replicate natural tones",
        "Highlighted by the controlled ambiance of artificial lighting",
    ],
    'Side Lighting': [
        "With one side of the face distinctly lit, casting a dramatic play of shadows",
        "Highlighted predominantly from one side, creating a depth of contrasts",
        "Illuminated from the side, accentuating textures and contours",
        "Illuminated from the side, creating stark contrasts of light and shadow",
        "With one side of the face brightly lit and the other in deep shadow",
        "Defined by the dramatic interplay of light and shadow from the side",
    ],
    'Loop Lighting': [
        "With a soft shadow gracing the cheek from loop lighting",
        "Characterized by the distinct loop shadow on the cheek",
        "Illuminated with the classic loop lighting technique, casting a subtle shadow",
        "Characterized by the small shadow of the nose creating a loop on the cheek",
        "Highlighted with a loop lighting technique, casting subtle shadows",
        "With the signature loop shadow adding depth and dimension",
    ],
    'Butterfly Lighting': [
        "With a soft shadow under the nose reminiscent of a butterfly",
        "Illuminated from above, casting a butterfly-like shadow",
        "With the characteristic shadow of butterfly lighting under the nose",
        "Defined by the butterfly-shaped shadow under the nose",
        "Illuminated with the classic butterfly lighting technique, creating an elegant effect",
        "Bathed in light from above, casting a distinct butterfly shadow",
    ],
    'Backlighting': [
        "With light emanating from the back, creating a halo effect",
        "Highlighted from behind, casting a gentle glow around the silhouette",
        "Illuminated predominantly from the back, emphasizing the contours",
    ],
    'Soft/Diffused Lighting': [
        "Bathed in the gentle, diffused light that softens features",
        "Under the soft glow that evenly illuminates without harsh shadows",
        "Illuminated with a diffused light, creating a dreamy ambiance",
    ],
    'Golden Hour': [
        "Basked in the warm, golden tones of the hour before sunset",
        "Golden hour light bathes the scene, casting a warm glow and elongating shadows",
        "Illuminated by the magical soft light of the golden hour",
        "Under the enchanting, warm glow of the golden hour",
    ],
    'Blue Hour': [
        "Surrounded by the cool, ethereal tones of the blue hour",
        "Illuminated by the serene, bluish hue of the hour after sunset",
        "Under the mystical, twilight ambiance of the blue hour",
    ],
    'Split Lighting': [
        "With half the face brightly lit and the other in shadow, creating a split effect",
        "Defined by the dramatic contrast of light and shadow on the face",
        "Illuminated with a split lighting technique, emphasizing duality",
    ],
    'Rembrandt Lighting': [
        "Characterized by the distinct triangle of light on the cheek",
        "Illuminated in the classic Rembrandt style, balancing light and shadow",
        "With the signature Rembrandt triangle gracing the face",
    ],
    'Top-Down Lighting': [
        "Illuminated from above, casting definitive shadows below features",
        "The overhead fluorescent lights cast a cool, even tone over the scene, adding a modern urban vibe",
        "With light source directly overhead, emphasizing depth",
        "Bathed in the light coming from the top, creating a theatrical effect",
    ],
    'Broad Lighting': [
        "With the face predominantly lit on its broadest side, emphasizing width",
        "Highlighted with broad lighting, casting minimal shadows",
        "Illuminated in a manner that enhances the face's broader perspective",
    ],
    'Short Lighting': [
        "With the face illuminated from its narrow side, emphasizing depth and contour",
        "Defined by the short lighting technique, adding drama and intensity",
        "Highlighted in a way that captures the face's contours and depth",
    ],
    'Flash Lighting': [
        "Illuminated with a sharp burst of light, capturing vivid details",
        "Defined by the sudden, bright illumination from a flash",
        "With features crisply lit by the distinct light of a flash",
    ],
    'Ambient Lighting': [
        "Surrounded by the soft, even tones of ambient light",
        "Basked in the gentle and consistent glow of ambient lighting",
        "With the nuances highlighted by the surrounding ambient light",
    ],
    'Directional Lighting': [
        "Illuminated with light coming from a specific direction, emphasizing depth",
        "Defined by the strong, unidirectional light source",
        "With shadows and highlights created by a clear directional light",
    ],
    'Fill Lighting': [
        "Balanced with fill light to soften shadows and even out contrasts",
        "With features gently lit by fill lighting, reducing harshness",
        "Softened by the subtle effects of fill light, creating harmony",
    ],
    'High Key Lighting': [
        "Surrounded by an abundance of bright light, reducing harsh shadows",
        "With features softly lit by high key lighting, evoking an airy atmosphere",
        "Illuminated in a manner that minimizes contrast and shadow, typical of high key lighting",
    ],
    'Low Key Lighting': [
        "Engulfed in deep shadows and minimal light, creating a moody ambiance",
        "Defined by the stark contrast and drama of low key lighting",
        "With features accentuated by the intense interplay of light and dark characteristic of low key lighting",
    ],
    'Motivated Lighting': [
        "Illuminated in a way that feels organic and inspired by elements within the scene",
        "With lighting that seems naturally sourced from items in the environment",
        "Highlighted by light that appears to have a clear, believable source within the context",
    ],
    'Practical Lighting': [
        "Lit by visible light sources present in the scene like lamps or candles",
        "With the warm and genuine glow from practical lights setting the tone",
        "Bathed in the authentic luminescence of actual light fixtures within the shot",
    ],
    'Bounced Lighting': [
        "Softly illuminated by light that's been reflected off surfaces, reducing harshness",
        "With a gentle and even glow resulting from bounced light",
        "Highlighted by the diffused and broadened effect of light that's been redirected",
    ],
    'Hard Lighting': [
        "Defined by the sharp shadows and bright highlights of direct light",
        "With features crisply lit by a focused light source, creating strong contrasts",
        "Illuminated in a manner that emphasizes texture and form through hard light",
    ],
    'Three-Point Lighting': [
        "Illuminated with the classic three-point setup, balancing key, fill, and back lights",
        "Defined by the harmony of three-point lighting, creating depth and texture",
        "With key, fill, and back lights working in concert to create a rich visual experience",
        "Highlighted by the versatility of three-point lighting, offering a balanced look",
    ],
    'Flat Lighting': [
        "Bathed in flat lighting that minimizes shadows",
        "With even illumination across the face, characteristic of flat lighting",
        "Illuminated in a manner that reduces shadow and contrast, typical of flat lighting",
        "Highlighted by the soft and shadowless effect of flat lighting",
    ],
    'Rim Lighting': [
        "With the edges softly outlined by rim lighting",
        "Illuminated from behind, creating a distinct rim of light",
        "Defined by the ethereal outline created by rim lighting",
        "Highlighted by the radiant border of rim lighting",
    ],
    'Clamshell Lighting': [
        "With features softly lit by the dual glow of clamshell lighting",
        "Defined by the flattering, even light of a clamshell setup",
        "Illuminated with clamshell lighting, producing soft and beauty-enhancing effects",
        "Highlighted by the glamour-inducing clamshell lighting technique",
    ],
    'Cross Lighting': [
        "Illuminated by lights from opposite sides, creating dynamic contrasts",
        "With textures and dimensions emphasized by the effects of cross lighting",
        "Defined by the interplay of dual light sources in a cross lighting setup",
        "With features enriched by the opposing forces of cross lighting",
    ],
    'Kicker Lighting': [
        "With a subtle highlight along the edge from kicker lighting",
        "Illuminated by a low-angle kicker light, adding depth",
        "Defined by the accentuating edge glow of kicker lighting",
        "Highlighted by the low and side-angled kicker light",
    ],
    'Cinematic Lighting': [
        "Illuminated in a cinematic style, evoking mood and atmosphere",
        "With the dramatic flair commonly found in cinematic lighting setups",
        "Defined by the atmospheric depth of cinematic lighting",
        "Surrounded by the emotional ambiance characteristic of cinematic lighting",
    ],
    'Stage Lighting': [
        "Lit with the broad and dynamic range of stage lighting",
        "With features emphasized by theatrical stage lights",
        "Defined by the vibrant and dramatic nature of stage lighting",
        "Bathed in the spotlight, typical of stage lighting setups",
    ],
    'Beauty Dish Lighting': [
        "With features softly lit by the focused glow of a beauty dish",
        "Illuminated by the flattering and directional light of a beauty dish",
        "Defined by the unique soft yet focused light of a beauty dish",
        "Highlighted by the beauty-enhancing qualities of beauty dish lighting",
    ],
    'Tungsten Lighting': [
        "Bathed in the warm, yellow-orange glow of tungsten lighting",
        "Illuminated by the classic, warm tones of a tungsten light source",
        "With features highlighted by the cozy atmosphere created by tungsten lighting",
        "Defined by the nostalgic and warm feel of tungsten lighting",
    ],
}

ethnicities_dict = {
    'European': [
        'Austrian', 'Portuguese', 'Russian', 'German', 'French', 'English', 'Swedish', 'Danish', 'Norwegian', 
        'Polish', 'Lithuanian', 'Hungarian', 'Italian', 'Spanish', 'Irish', 'Greek', 'Canadian', 'Romanian', 
        'Serbian', 'Croatian', 'Belgian', 'Icelandic', 'Swiss', 'Luxembourgish', 'Maltese', 'Andorran', 
        'Monacan', 'Liechtensteiner', 'San Marinese', 'Vatican', 'Maltese', 'Finnish', 'Latvian', 'Estonian', 
        'Macedonian', 'Albanian', 'Bosnian', 'Kosovar', 'Montenegrin', 'Moldovan', 'Bulgarian', 'Czech', 
        'Slovak', 'Armenian', 'Azerbaijani', 'Belarusian', 'Ukrainian', 'British', 'Dutch', 'Slovenian', 
        'Estonian', 'Cypriot'
    ],
    'Sub-Saharan African': [
        'Nigerian', 'Kenyan', 'Ghanaian', 'Ethiopian', 'South African', 'Congolese', 'Somali', 'Ugandan',
        'Tanzanian', 'Rwandan', 'Malawian', 'Zambian', 'Zimbabwean', 'Angolan', 'Botswanan', 'Madagascan', 'Gabonese',
        'Namibian', 'Senegalese', 'Cameroonian', 'Ivorian', 'Guinean', 'Liberian', 'Sierra Leonean', 'Burkinabe',
        'Malian', 'Togolese', 'Beninese', 'Nigerien', 'Chadian', 'Central African', 'South Sudanese', 'Eritrean', 'Djiboutian',
        'Comoran', 'Seychellois', 'Mauritian', 'Cape Verdean'
    ],
    'Middle Eastern': [
        'Israeli', 'Iraqi', 'Egyptian', 'Iranian', 'Afghan', 'Arab', 'Turkish', 'Persian', 'Georgian', 
        'Yemeni', 'Saudi Arabian', 'Lybian', 'Jordanian', 'Syrian', 'Lebanese', 'Omani', 'Palestinian', 
        'Qatari', 'Emirati', 'Bahraini', 'Kuwaiti', 'Azerbaijani', 'Armenian'
    ],
    'Latin American': [
        'Brazilian', 'Mexican', 'Argentinian', 'Colombian', 'Peruvian', 'Chilean', 'Ecuadorian', 'Bolivian', 
        'Venezuelan', 'Uruguayan', 'Paraguayan', 'Costa Rican', 'Panamanian', 'Nicaraguan', 'Guatemalan', 
        'Salvadoran', 'Honduran', 'Cuban', 'Dominican', 'Puerto Rican'
    ],
    'Oceanian': [
        'Australian', 'New Zealander', 'Fijian', 'Samoan', 'Tongan', 'Papuan', 'Guamanian', 'Palauan', 
        'Micronesian', 'Marshallese', 'Nauruan', 'Solomon Islander', 'Vanuatuan', 'Ni-Vanuatu', 'New Caledonian', 
        'French Polynesian', 'Tokelauan', 'Tuvaluan', 'Wallisian', 'Futunan'
    ],
    'Caribbean': [
        'Cuban', 'Jamaican', 'Haitian', 'Dominican', 'Trinidadian', 'Barbadian', 'Bahamian', 'Grenadian', 
        'Saint Lucian', 'Antiguan', 'Vincentian', 'Kittitian', 'Nevisian', 'Montserratian', 'Bermudian'
    ],
    'Central Asian': [
        'Uzbek', 'Kazakh', 'Tajik', 'Turkmen', 'Kyrgyz', 'Uzbekistani', 'Turkistani', 'Uyghur', 'Tatar', 
        'Karakalpak', 'Bashkir', 'Kumyk', 'Balkar', 'Karachay', 'Avar'
    ],
    'West Asian': [
        'Armenian', 'Azerbaijani', 'Georgian', 'Turkish', 'Kurdish', 'Assyrian', 'Alevi', 'Druze', 'Yazidi', 
        'Maronite', 'Alawite', 'Circassian', 'Laz', 'Gilaki', 'Mazandarani'
    ],
    'North African': [
        'Egyptian', 'Moroccan', 'Algerian', 'Tunisian', 'Libyan', 'Sudanese', 'Mauritanian', 'Berber', 'Amazigh', 
        'Nubian', 'Coptic', 'Tuareg', 'Riffian', 'Kabyle', 'Sahrawi'
    ],
    'Scandinavian': [
        'Swedish', 'Norwegian', 'Danish', 'Icelandic', 'Finnish', 'Sami', 'Faroese', 'Gotlander', 'Orcadian', 
        'Shetlander', 'Ålandic', 'Jämtlander'
    ],
    'North American': [
        'American', 'Canadian', 'Mexican', 'Greenlandic', 'Alaskan', 'Texan', 'Quebecois', 'Cajun', 'Hawaiian', 
        'Newfoundlander', 'Cree', 'Inuvialuit', 'Métis', 'Gwich’in', 'Haida', 'Tlingit'
    ],
    'Arctic': [
        'Inuit', 'Saami', 'Chukchi', 'Yupik', 'Aleut', 'Kalaallit', 'Nenets', 'Khanty', 'Evenki', 'Selkup', 
        'Yamalo', 'Enets', 'Nganasan', 'Veps', 'Koryaks'
    ],
    'Southeast Asian': [
        'Thai', 'Laos', 'Cambodian', 'Malaysian', 'Filipino', 'Indonesian', 'Burmese', 'Singaporean', 
        'Vietnamese', 'Bruneian', 'Timorese', 'Javanese', 'Balinese', 'Sundanese', 'Minangkabau'
    ],
    'Balkan': [
        'Bulgarian', 'Greek', 'Romanian', 'Serbian', 'Croatian', 'Bosnian', 'Slovenian', 'Montenegrin', 
        'Macedonian', 'Albanian', 'Kosovar', 'Vlach', 'Pomak', 'Torlakian', 'Gagauz', 'Aromanian'
    ],
    'Polynesian': [
        'Hawaiian', 'Maori', 'Samoan', 'Tongan', 'Tahitian', 'Niuean', 'Tokelauan', 'Tuvaluan', 'Rapanui',
        'Marquesan', 'Mangarevan', 'Pukapukan', 'Rarotongan', 'Tahitian', 'Tuamotuan', 'Rennell Islander'
    ],
    'Micronesian': [
        'Guamanian', 'Palauan', 'Marshallese', 'Nauruan', 'Micronesian', 'Kosraean', 'Yapese', 'Pohnpeian', 
        'Chuukese', 'Angauran', 'Sonsorolese', 'Tobi Islander', 'Woleaian', 'Ulithian', 'Carolinian'
    ],
    'Melanesian': [
        'Fijian', 'Papuan', 'Vanuatuan', 'Solomon Islander', 'Ni-Vanuatu', 'New Caledonian', 'Kanak', 'Bougainvillean',
        'Ambrym Islander', 'Aniwa Islander', 'Futuna Islander', 'Erromango Islander', 'Tannese', 'Motu', 'Tolai'
    ],
    'Indigenous American': [
        'Navajo', 'Mayan', 'Inca', 'Guarani', 'Mapuche', 'Quechua', 'Aymara', 'Taino', 'Kuna', 'Wayuu',
        'Cherokee', 'Lakota', 'Apache', 'Iroquois', 'Zapotec', 'Mixtec', 'Quechuan'
    ],
    'Australasian': [
        'Australian', 'New Zealander', 'Papuan', 'Melanesian', 'Polynesian', 'Micronesian',
        'Torres Strait Islander', 'Tiwi Islander', 'Anangu', 'Noongar', 'Palawa', 'Yolngu', 'Koori'
    ],
    'Caucasian': [
        'Georgian', 'Chechen', 'Dagestani', 'Armenian', 'Azerbaijani', 'Abkhaz', 'Ossetian', 'Circassian',
        'Ingush', 'Kabardian', 'Balkar', 'Karachay', 'Lezgian', 'Aghul', 'Tabasaran'
    ],
    'East Asian': [
        'Chinese', 'Vietnamese', 'Japanese', 'Korean', 'Mongolian', 'Taiwanese',
        'Hong Kongese', 'Macanese', 'Ryukyuan', 'Ainu', 'Hui', 'Uighur'
    ],
    'South Asian': [
        'Indian', 'Pakistani', 'Bangladeshi', 'Sri Lankan', 'Nepalese', 'Bhutanese', 'Maldivian',
        'Sinhalese', 'Tamil', 'Pashtun', 'Sindhi', 'Punjabi', 'Gujarati'
    ],
}

eye_colors_dict = {
    "European": [
        "blue", "green", "gray", "hazel", "brown", "amber", "ice-blue", "steel gray",
        "sea green", "pale blue", "deep blue", "emerald green", "light brown", "dark brown",
        "grey-blue", "hazel-green", "turquoise", "aqua", "violet", "olive"
    ],
    "Sub-Saharan African": [
        "dark brown", "black", "amber", "deep brown", "onyx", "chocolate brown",
        "copper", "golden", "honey colored"
    ],
    "Middle Eastern": [
        "brown", "dark brown", "black", "hazel", "amber", "green", "blue",
        "honey colored", "golden", "hazel-green"
    ],
    "Latin American": [
        "brown", "dark brown", "black", "hazel", "amber", "green", "blue", "gray",
        "honey colored", "golden", "copper"
    ],
    "Oceanian": [
        "dark brown", "black", "light brown", "amber", "hazel", "deep brown"
    ],
    "Caribbean": [
        "brown", "dark brown", "black", "hazel", "amber", "green", "honey colored"
    ],
    "Central Asian": [
        "brown", "dark brown", "black", "hazel", "amber", "green", "blue", "gray"
    ],
    "West Asian": [
        "brown", "dark brown", "black", "hazel", "amber", "green", "blue", "gray"
    ],
    "North African": [
        "brown", "dark brown", "black", "hazel", "amber", "green", "honey colored"
    ],
    "Scandinavian": [
        "blue", "ice-blue", "gray", "green", "pale blue", "light brown", "icy grey",
        "sea green", "turquoise", "aqua"
    ],
    "North American": [
        "blue", "green", "gray", "hazel", "brown", "amber", "dark brown", "black",
        "hazel-green", "light brown", "honey colored"
    ],
    "Arctic": [
        "dark brown", "black", "hazel", "deep brown"
    ],
    "Southeast Asian": [
        "dark brown", "black", "light brown", "amber", "hazel", "copper"
    ],
    "Balkan": [
        "brown", "dark brown", "black", "hazel", "green", "blue", "gray", "amber"
    ],
    "Polynesian": [
        "dark brown", "black", "light brown", "amber", "deep brown"
    ],
    "Micronesian": [
        "dark brown", "black", "light brown", "deep brown"
    ],
    "Melanesian": [
        "dark brown", "black", "light brown", "amber", "deep brown"
    ],
    "Indigenous American": [
        "dark brown", "black", "light brown", "amber", "hazel", "copper"
    ],
    "Australasian": [
        "dark brown", "black", "light brown", "amber", "hazel", "blue", "green",
        "gray", "honey colored"
    ],
    "Caucasian": [
        "brown", "dark brown", "black", "hazel", "green", "blue", "gray", "amber",
        "hazel-green", "light brown"
    ],
    "East Asian": [
        "dark brown", "black", "light brown", "amber", "copper"
    ],
    "South Asian": [
        "dark brown", "black", "light brown", "amber", "hazel", "copper", "honey colored"
    ],
    "Interesting Colors": [
        'blue', 'green', 'teal', 'hazel', 'brown', 'amber', 'ice-blue', 'steel gray',
        'cyan', 'violet', 'red', 'pink', 'orange', 'yellow', 'gold', 'silver', 'black',
        'white', 'gray', 'purple', 'turquoise', 'aqua', 'emerald', 'sapphire', 'ruby',
        'topaz', 'amethyst', 'jade', 'opal', 'pearl', 'sable', 'copper', 'bronze', 'brass',
        'platinum', 'rose gold', 'champagne', 'mahogany', 'caramel', 'cinnamon', 'coffee'
    ]
}

hair_colors_dict = {
    "European": [
        "blond", "light brown", "dark brown", "black", "red", "auburn", "strawberry blond", "platinum blond",
        "ash blond", "dirty blond", "golden blond", "chestnut", "mahogany", "copper", "ginger",
        "silver-gray", "white", "raven-black", "jet black", "honey blond", "sandy"
    ],
    "Sub-Saharan African": [
        "black", "dark brown", "brown", "reddish-brown", "auburn", "jet black",
        "ebony", "chocolate brown", "espresso"
    ],
    "Middle Eastern": [
        "black", "dark brown", "brown", "light brown", "auburn", "jet black", "chestnut brown"
    ],
    "Latin American": [
        "black", "dark brown", "brown", "light brown", "auburn", "reddish-brown",
        "jet black", "mahogany", "chestnut brown"
    ],
    "Oceanian": [
        "black", "dark brown", "brown", "jet black", "ebony"
    ],
    "Caribbean": [
        "black", "dark brown", "brown", "reddish-brown", "auburn", "jet black", "ebony"
    ],
    "Central Asian": [
        "black", "dark brown", "brown", "light brown", "jet black"
    ],
    "West Asian": [
        "black", "dark brown", "brown", "light brown", "auburn", "jet black"
    ],
    "North African": [
        "black", "dark brown", "brown", "light brown", "jet black", "ebony"
    ],
    "Scandinavian": [
        "blond", "light brown", "dark brown", "platinum blond", "ash blond", "golden blond",
        "strawberry blond", "silver-gray", "white"
    ],
    "North American": [
        "blond", "light brown", "dark brown", "black", "red", "auburn", "strawberry blond", "platinum blond",
        "ash blond", "dirty blond", "golden blond", "chestnut", "mahogany", "copper", "ginger",
        "silver-gray", "white", "raven-black", "jet black", "honey blond", "sandy"
    ],
    "Arctic": [
        "black", "dark brown", "jet black"
    ],
    "Southeast Asian": [
        "black", "dark brown", "brown", "jet black", "ebony"
    ],
    "Balkan": [
        "dark brown", "black", "light brown", "auburn", "chestnut brown", "mahogany"
    ],
    "Polynesian": [
        "black", "dark brown", "brown", "jet black", "ebony"
    ],
    "Micronesian": [
        "black", "dark brown", "jet black", "ebony"
    ],
    "Melanesian": [
        "black", "dark brown", "reddish-brown", "jet black", "ebony"
    ],
    "Indigenous American": [
        "black", "dark brown", "brown", "reddish-brown", "jet black", "ebony"
    ],
    "Australasian": [
        "black", "dark brown", "brown", "blond", "red", "auburn", "jet black", "ebony"
    ],
    "Caucasian": [
        "dark brown", "black", "light brown", "auburn", "red", "chestnut brown",
        "mahogany", "jet black"
    ],
    "East Asian": [
        "black", "dark brown", "light brown", "auburn", "reddish-brown", "jet black", "ebony"
    ],
    "South Asian": [
        "black", "dark brown", "brown", "auburn", "reddish-brown", "jet black", "ebony"
    ],
    "Interesting Colors": [
        'blue', 'green', 'teal', 'hazel', 'brown', 'amber', 'ice-blue', 'steel gray',
        'cyan', 'violet', 'red', 'pink', 'orange', 'yellow', 'gold', 'silver', 'black',
        'white', 'gray', 'purple', 'turquoise', 'aqua', 'emerald', 'sapphire', 'ruby',
        'topaz', 'amethyst', 'jade', 'opal', 'pearl', 'sable', 'copper', 'bronze', 'brass',
        'platinum', 'rose gold', 'champagne', 'mahogany', 'caramel', 'cinnamon', 'coffee'
    ]
}

expressions_list = [
    'with a fearful expression', 'with a whimsical twirl', 'with a contemptuous sneer', 'with an eager nod',
    'with a sulky expression', 'with a concerned expression', 'with a wistful expression', 'with tongue sticking out',
    'with a elated expression', 'with a nostalgic expression', 'with a guilty expression', 'with a optimistic expression',
    'with a bored expression', 'with a uninterested expression', 'with a perplexed scratch', 'with a bewildered expression',
    'with an excited expression', 'with a serene expression', 'with a stunned expression', 'with a vigilant watch',
    'with a grateful expression', 'with a sad gaze', 'with a laughing expression', 'with a cautious approach',
    'with a relieved sigh', 'with a sorrowful expression', 'with a scared expression', 'with a cheerful expression',
    'with an inquisitive tilt', 'with a scary expression', 'with a distressed expression', 'with a angry expression',
    'with a proud expression', 'with a resentful expression', 'with a timid expression', 'with a frustrated expression',
    'with a surprised look', 'with a inquisitive expression', 'with a intense expression', 'with a hollow stare',
    'with a solemn expression', 'with a gloomy stare', 'with a zany expression', 'with a contemplative expression',
    'with a jealous glance', 'with an ecstatic cheer', 'with a pleased expression', 'with an enraged expression',
    'with a hollow stare expression', 'with a irritated expression', 'with a humbled bow', 'with an appreciative nod',
    'with a joyful smile', 'with a astonished expression', 'with a satisfied expression', 'with a worried frown',
    'with a playful smile', 'with a melancholy look', 'with a mournful cry', 'with a determined stride',
    'with a lonely expression', 'with a jubilant dance', 'with a annoyed expression', 'with a disgusted expression',
    'with a amused expression', 'with a humbled expression', 'with a triumphant expression', 'with a ecstatic expression',
    'with a dreamy expression', 'with a blank stare expression', 'with a nervous expression', 'with a panic expression',
    'with a disgruntled expression', 'with a shy expression', 'with a determined expression', 'with a pensive look',
    'with a melancholy expression', 'with a longing gaze', 'with a regretful sigh', 'with a grumpy grunt',
    'with a skeptical eyebrow', 'with a focused expression', 'with a indignant expression', 'with a confused look',
    'with a calm expression', 'with a stoic expression', 'with an annoyed grimace', 'with an arrogant posture',
    'with a wary eye', 'with an uneasy shuffle', 'with a disbelief expression', 'with an anxious fidget',
    'with a longing expression', 'with an overwhelmed gasp', 'with mouth open in surprise', 'with a puzzled expression',
    'with a joyful expression', 'with a hysteric expression', 'with a anxious expression', 'with a embarrassed expression',
    'with a confused expression', 'with a satisfied smile', 'with a resigned shrug', 'with a hysterical laugh',
    'with a curious glance', 'with a apprehensive expression', 'with a hopeful expression', 'with a determined look',
    'with a crying expression', 'with a mad expression', 'with a whimsical expression', 'with a vexed expression',
    'with a sly smile', 'with an optimistic smile', 'with a zany hop', 'with a relieved expression',
    'with a smiling expression', 'with a indifferent expression', 'with a mournful expression', 'with a shocked expression',
    'with a serious expression', 'with a pessimistic mutter', 'with a vexed stomp', 'with a awkward expression',
    'with a thoughtful gaze', 'with a frowning expression', 'with a frustrated gesture', 'with a beaming expression',
    'with tears in the eyes', 'with a grinning expression', 'with a jealous expression', 'with a fearful look',
    'with a deflated posture', 'with a grieving expression', 'with a confident expression', 'with a depressed expression',
    'with an amused smirk', 'with a pleased nod', 'with an enthusiastic expression', 'with a overwhelmed expression',
    'with a frown', 'with a irate expression', 'with a gloomy expression', 'with a hopeful gaze',
    'with a content expression', 'with a thoughtful expression', 'with a cheerful demeanor', 'with an apprehensive gaze',
    'with a heartbroken expression', 'with a detached expression', 'with a disoriented expression', 'with a tormented expression',
    'with an awkward smile', 'with a timid step', 'with a bewildered look', 'with a wide smile',
    'with a uneasy expression', 'with a content smile', 'with a bored yawn', 'with a disappointed expression',
    'with a appreciative expression', 'with a furious expression', 'with an elated jump', 'with a enthusiastic expression',
    'with a pensive expression', 'with a resigned expression', 'with a panicked expression', 'with a sulky pout',
    'with a tense posture', 'with a melancholic expression', 'with a excited expression', 'with an angry look',
    'with a sad expression', 'with a resentful glare', 'with a disappointed frown', 'with a apathetic expression',
    'with a proud stance', 'with a beaming smile', 'with a grumpy expression', 'with a frowning face',
    'with an indignant huff', 'with an inquisitive look', 'with a eager expression', 'with a stressed expression',
    'with a funny expression', 'with a triumphant roar', 'with a skeptical expression', 'with an apathetic stare',
    'with an irate shout', 'with a worried expression', 'with a cautious expression', 'with a disgruntled scowl',
    'with a delighted expression', 'with a scowl', 'with a happy demeanor', 'with a curious expression',
    'with an indifferent shrug', 'with a deflated expression', 'with an embarrassed blush', 'with a distressed cry',
    'with a surprised expression', 'with a horrified look', 'with a perplexed expression', 'with a look of disbelief',
    'with a contempt expression', 'with a lonely look', 'with a blank stare', 'with a exasperated expression',
    'with a interested expression', 'with a jubilant expression', 'with a vigilant expression', 'with a flabbergasted expression',
    'with a regretful expression', 'with a neutral expression', 'with an admirable expression', 'with a nervous twitch',
    'with a grateful smile', 'with an exasperated sigh', 'with a ashamed expression', 'with a enraged expression',
    'with a disgusted look', 'with a solemn face', 'with a pessimistic expression', 'with a calm demeanor',
    'with a wary expression', 'with a admirable expression', 'with a baffled expression', 'with a wistful glance',
    'with a horrified expression', 'with a tense expression', 'with a joyous laugh', 'with an interested look',
    'with a mischievous grin', 'with an ashamed face', 'with a arrogant expression', 'with a scary look', 'with a happy expression'
]

hair_styles_list = [
    "flowing hair", "short curly hair", "bald head", "wavy hair", "short spiky hair",
    "long straight hair", "short straight hair", "long curly hair", "shoulder-length hair", "tidy hair",
    "shaven head", "buzz cut hair", "bob cut hair", "afro hair", "dreadlocks",
    "braided hair", "ponytail hair", "hair bun", "shiny hair", "mullet hair",
    "pixie cut hair", "undercut hair", "fade haircut", "taper haircut", "quiff hair",
    "faux hawk hair", "pompadour hair", "wet hair look", "crew cut hair", "side-parted hair",
    "mohawk hair", "comb over hair", "slicked-back hair", "shaggy hair", "layered hair",
    "choppy hair", "asymmetrical haircut", "feathered hair", "cropped hair", "blunt cut hair",
    "razor cut hair", "textured hair", "coiled hair", "ringlet hair", "cornrows hair",
    "finger-waved hair", "pin curled hair", "beehive hair", "pageboy haircut", "hime cut hair",
    "pixie-bob haircut", "lob haircut", "jheri curl hair", "curtain hair", "top knot hair",
    "man bun hair", "twisted hair", "locs hair", "permed hair", "hair with bangs",
    "hair with fringe", "balayage hair", "hair with highlights", "hair with lowlights", "hair with chunky highlights",
    "hair with frosted tips", "medium-length hair", "coily hair", "side-part hair", "middle-part hair",
    "twist-out hair", "bantu knots hair", "box braids hair", "goddess braids hair", "faux locs hair",
    "twist braids hair", "finger coils hair", "perm rod set hair", "sidecut hair", "wolf cut hair",
    "curtain bangs hair", "baby bangs hair", "side-swept bangs hair", "ducktail hair", "liberty spikes hair",
    "deathhawk hair", "French twist hair", "chignon hair", "bouffant hair", "victory rolls hair",
    "crown braid hair", "milkmaid braids hair", "space buns hair", "ombre hair", "dip-dyed hair",
    "streaked hair", "colorful hair", "pastel-colored hair", "neon-colored hair", "Gibson Girl hair",
    "hi-top fade hair", "flattop hair", "Rockabilly hair", "Teddy Boy hair", "Mod hair", "hair adorned with flowers",
    "Hippie hair", "sculptural hair", "geometric hair", "futuristic hair", "avant-garde hair",
    "editorial hair", "haute couture hair", "gravity-defying hair", "buzz cut with designs", "tapered fade hair",
    "high and tight hair", "textured top hair", "long on top, short on sides hair", "chin-length hair", "collarbone-length hair",
    "mid-back length hair", "waist-length hair", "half-up, half-down hair", "side-braided hair", "braided mohawk hair",
    "faux hawk with braided sides", "twisted updo hair", "messy bun with loose tendrils", "sleek ponytail with baby hairs", 
    "hair with jewelry", "hair with ornate pins", "hair with a headband", "hair with a wrap", "hair with colorful extensions"
]

face_poses_list = [
    "facing directly at the camera", "with a slight turn to the left", "with a slight turn to the right",
    "in three-quarter view to the left", "in three-quarter view to the right", "in full left profile",
    "in full right profile", "looking up at the camera", "looking down at the camera",
    "with chin slightly raised", "with chin slightly lowered", "with head tilted to the left",
    "with head tilted to the right", "with a subtle lean forward", "with a subtle lean backward",
    "with shoulders at an angle", "with head slightly rotated left", "with head slightly rotated right",
    "in side-profile view", "with head leaning to the left", "with head leaning to the right",
    "with chin jutted out", "with chin tucked in", "with head tilted back",
    "with head tilted forward", "at a low angle to the camera", "at a high angle to the camera",
    "facing away from the camera", "looking over left shoulder", "looking over right shoulder",
    "with head cocked to the left", "with head cocked to the right", "at eye level with the camera",
    "camera slightly below eye level", "camera slightly above eye level", "facing the camera",
    "in profile", "looking away", "in three-quarter view", "looking up", "looking down", "with head tilted to one side",
    "with eyes looking off-camera", "tilting their head slightly to the right", "tilting their head slightly to the left",
    "with a straight head position", "with their chin lifted slightly", "with their chin lowered a bit",
    "in a left profile view", "in a right profile view", "with their head slightly tilted back",
    "with their head leaning forward", "with a slight three-quarter view to the right", "with a slight three-quarter view to the left",
    "with their head leaning to the right", "with their head leaning to the left", "with their head slightly rotated to the right",
    "with their head slightly rotated to the left", "with their chin slightly jutted out", "with their chin slightly tucked in",
    "with a three-quarter view of their face", "with a neutral head position", "with their head slightly tilted to the left",
    "with their head tilted back a little", "with their head leaning slightly forward", "with their head rotated a bit to the right",
    "with their head rotated a bit to the left", "with their chin pointing slightly upwards", "with their face resting on their hand"
]

#%% helper functions

def get_prompt_start():
    prompt_start_list = [
        "Photo of a", "Portrait of a", "Photograph of a", "Medium Shot of a", "Close-Up of a", "An artistic portrayal of a",
        "Headshot of a", "Face of a", "Facial portrait of a", "Studio portrait of a", "Candid portrait of a", "A serene image of a",
        "Profile view of a", "Character study of a", "Expressive portrait of a", "Cinematic portrait of a", "A candid portrait of a",
        "A portrait photo of a", "A professional photograph of a", "A professional portrait photograph of a", "A pro portrait photo of a", 
        "A high-resolution image of a", "A captivating picture of a", "An enchanting photo of a", "A studio shot of a", "A casual snapshot of a",
        "A meticulously composed portrait of a", "An authentic picture of a", "A magazine-quality portrait of a", "A compelling photograph of a",
        "A striking portrait of a", "A vintage photograph of a", "A black and white portrait of a", "An evocative image of a", 
        "A low-angle shot of a", "A wide-angle shot of a", "An atmospheric portrait of a", "A moody portrayal of a", "A whimsical image of a",
        "An extreme wide shot of a", "A wide shot of a", "A full shot of a", "A medium wide shot of a", "A medium close-up of a",
        "An extreme close-up of a", "An eye-level shot of a", "A Dutch angle shot of a", "A tracking shot of a", "A pan shot of a",
        "A tilt shot of a", "A dolly shot of a", "A zoom shot of a", "An over-the-shoulder shot of a", "A POV shot of a",
        "A cutaway shot of a", "An insert shot of a", "An aerial portrait shot of a", "A high-angle shot of a",
    ]

    return random.choice(prompt_start_list)

def get_random_glasses():
    glasses_list = [
        "wearing classic rectangular glasses", "with round vintage-style glasses", "sporting cat-eye frames",
        "with aviator-style glasses", "wearing oversized square glasses", "with sleek rimless glasses",
        "sporting horn-rimmed glasses", "with retro browline glasses", "wearing geometric hexagonal frames",
        "with trendy clear frame glasses", "sporting thick-framed hipster glasses", "with oval wire-frame glasses",
        "wearing sporty wraparound glasses", "with stylish half-rim glasses", "sporting colorful acetate frames",
        "with sophisticated titanium frames", "wearing bold colored glasses", "with minimalist thin metal frames",
        "sporting funky asymmetrical glasses", "with classic wayfarers", "wearing trendy blue light blocking glasses",
        "with elegant gold-rimmed glasses", "sporting futuristic shield glasses", "with retro round sunglasses",
        "wearing clip-on sunglasses", "with gradient lens sunglasses", "sporting mirrored aviator sunglasses",
        "with polarized sports sunglasses", "wearing fashionable oversized sunglasses", "with classic clubmaster sunglasses",
        "sporting trendy transparent sunglasses", "with vintage cat-eye sunglasses", "wearing modern shield sunglasses",
        "with retro square sunglasses", "sporting stylish browline sunglasses", "with cool wrap-around sunglasses", 
        "with cyberpunk LED glasses", "wearing steampunk goggles", "sporting futuristic visor sunglasses",
        "with monocle", "wearing diamond-studded glasses", 
    ]
    
    return random.choice(glasses_list)

def get_random_gaze_direction():
    gaze_direction_list = [
        "looking directly into the camera", "gazing off to the side", 
        "looking down in thought", "looking upwards", "with eyes closed", 
        "staring to the side", "with a sidelong glance", "looking past the camera",
        "with a far-off look", "with a focused gaze", "with a wandering gaze",
        "looking into the distance", "with eyes nearly shut", 
        "with eyes fixed on an unseen object", "glancing over their shoulder",
        "with eyes darting around nervously", "staring intently at something off-camera",
        "with a thousand-yard stare", "looking through half-lidded eyes",
        "with eyes wide in surprise", "squinting against bright light",
        "with a dreamy, unfocused gaze", "looking down demurely",
        "with eyes crinkled in laughter", "peering curiously at the viewer",
        "with a piercing stare", "looking up through their lashes",
        "with eyes reflecting deep contemplation", "gazing longingly into the distance",
        "with a mischievous twinkle in their eyes", "looking sideways with suspicion",
        "with eyes brimming with tears", "staring defiantly at the camera",
        "with a vacant, expressionless gaze", "looking up in wonder",
        "with eyes narrowed in concentration", "gazing lovingly at someone off-camera",
        "with a faraway look of nostalgia", "looking down with a shy smile",
        "with eyes alight with excitement", "staring off into space pensively",
        "with a haunted look in their eyes", "glancing furtively to the side",
        "with eyes filled with determination", "looking straight ahead with resolve",
        "with a distant, melancholic gaze", "peering intently at something in their hands",
        "with eyes dancing with amusement", "staring blankly ahead",
        "with a wistful gaze towards the horizon", "looking down with a furrowed brow",
        "with eyes half-closed in contentment", "gazing upward with hope",
        "with a sharp, analytical stare", "looking sideways with skepticism",
        "with eyes wide with wonder", "staring intensely at their own reflection",
        "with a distant gaze, lost in memory", "looking directly at the viewer with vulnerability",
        "with eyes scanning the environment alertly", "gazing into middle distance, deep in thought",
        "with a penetrating stare that seems to see through the viewer", "looking down with eyes closed, in meditation",
        "with eyes darting back and forth, reading something", "staring off-camera with a look of longing",
        "with eyes widened in fear or shock", "gazing at their own hands with fascination",
        "with a soft, compassionate look in their eyes", "staring at the ground with a mix of shame and regret",
        "with eyes twinkling with inner joy", "looking past the camera with a stoic expression"
    ]

    return np.random.choice(gaze_direction_list)

def ger_facial_hair_description():
    facial_hair_list = [
        "clean-shaven", "with light stubble", "with heavy stubble",
        "with a short, neat beard", "with a full, thick beard", "with a long, flowing beard",
        "with a well-groomed goatee", "with a circle beard", "with a chin strap beard",
        "with a neat mustache", "with a handlebar mustache", "with a horseshoe mustache",
        "with mutton chops", "with friendly sideburns", "with a soul patch",
        "with a Van Dyke beard", "with a Garibaldi beard", "with a ducktail beard",
        "with a French fork beard", "with a Bandholz beard", "with a yeard",
        "with a ZZ Top-style beard", "with a 5 o'clock shadow", "with designer stubble",
        "with a pencil mustache", "with a Fu Manchu mustache", "with an imperial mustache",
        "with a Dali mustache", "with a walrus mustache", "with a chevron mustache",
        "with a Hollywoodian beard", "with a short boxed beard", "with a Verdi beard",
        "with a Spartan beard", "with a Norse beard", "with a Viking-style beard",
        "with a neatly trimmed beard", "with an unkempt beard", "with a patchy beard",
        "with a salt-and-pepper beard", "with a graying beard", "with a shabby chic beard",
        "with a faded beard", "with a tapered beard", "with a pointy beard",
        "with a braided beard", "with a forked beard", "with a sculpted beard",
        "with an Asian-style mustache", "with a handlebar-and-goatee combo",
        "with a thin-line beard", "with a disconnected mustache",
        "with a chin curtain beard", "with a Klingon-style beard",
        "with a wild, untamed beard", "with a precisely lined beard",
        "with a barely-there mustache", "with a bushy mustache",
        "with a curled mustache", "with waxed mustache tips",
        "with a scruffy beard", "with a lumberjack-style beard",
        "with a hipster beard", "with an artistically trimmed beard",
        "with a multi-colored dyed beard", "with a glitter beard",
        "with a freestyle beard", "with a neck beard",
        "with mutton chops connected to a mustache", "with a chin puff",
        "with an anchor beard", "with a Balbo beard", "with a royal beard", 
        "with a Zappa-style beard", "with a Hulihee beard",
        "with a long goatee", "with sideburns connected to a mustache",
        "with a mustache-free beard", "with a beard-free mustache",
        "with a pencil-thin chin strap", "with a double mustache",
        "with a triangle beard", "with an inverted T-shape beard",
    ]

    return random.choice(facial_hair_list)

def get_makeup_description():
    makeup_list = [
        "with natural, barely-there makeup", "wearing a classic red lip and winged eyeliner", "with a smoky eye and nude lips",
        "featuring a bold cat-eye and coral lipstick", "with a fresh, dewy look and pink blush", "wearing dramatic false eyelashes and glossy lips",
        "with a bronzed, sun-kissed glow", "featuring metallic eyeshadow and matte lips", "with a no-makeup makeup look",
        "wearing bold, colorful eyeshadow and neutral lips", "with perfectly contoured cheekbones", "featuring glossy eyelids and a subtle lip tint",
        "with a gothic-inspired dark lip and pale complexion", "wearing pastel eyeshadow and peach blush", "with a monochromatic makeup look in earthy tones",
        "featuring glitter accents around the eyes", "with a bold, avant-garde makeup design", "wearing a 1950s-inspired pin-up look",
        "with a subtle brown smoky eye and pink lips", "featuring holographic highlighter on cheekbones", "with minimal eye makeup and a bold berry lip",
        "wearing blue mascara and orange-tinted lips", "with graphic eyeliner designs", "featuring ombre lips from dark to light",
        "with strategically placed facial gems or rhinestones", "wearing an ethereal, fairy-like makeup look", "with a bold unibrow statement",
        "featuring neon eyeliner accents", "with a soft, romantic rose-gold palette", "wearing dramatic stage makeup with exaggerated features",
        "with a 1960s-inspired Twiggy lash look", "featuring bright, color-blocked eyeshadow", "with a glossy, wet-look eye makeup",
        "wearing a subtle everyday makeup with focus on skincare", "with an edgy, punk-inspired dark eye and bright lip", 
        "with artfully applied freckles", "wearing mermaid-inspired shimmery scales on cheekbones", "with a soft focus, blurred lip look",
        "featuring negative space eyeliner designs", "with an airbrushed, flawless complexion", "wearing ice princess-inspired frosty tones",
        "with a sun-striping technique using bronzer", "featuring floating crease liner", "with a soft focus hazy eye look",
        "wearing a classic French girl inspired minimal makeup", "with deconstructed bright eyeshadow placement", "featuring a cut-crease eyeshadow technique",
        "with an extreme contour and highlight", "wearing a watercolor-inspired soft wash of colors", "featuring a gradient lip from dark center to light edges",
    ]

    makeup_description = random.choice(makeup_list)
    return makeup_description

def get_location_setting_background():
    locations_settings_backgrounds_list = [
        "in a corn field", "in a wheat field", "in a rice field", "in a sunflower field", "in a strawberry field", 
        "in a lavender field", "in a tulip field", "in a pumpkin patch", "in a flower garden", "in a vegetable garden",
        "in a water garden", "in a rose garden", 'in a grass hill', 'in a grass field', 'in a grassy meadow', 'in a grassy plain',
        "at the desert", "in the forest", "in the park", "at the garden", "at the beach", "outside in wild nature",
        "at the lake", "outside near a mountain", "near a waterfall", "in a cherry blossom park",
        "on a snowy mountaintop", "in a botanical garden", "on a scenic cliff", "on a tropical island", "in a secluded cave",
        "in a vineyard", "in an orchard", "at a coral reef", "in a bamboo forest", "in an ice cave", "in a tulip field",
        "in a pumpkin patch", "next to a scenic pond", "in a tea plantation", "in a coffee plantation", "in an olive grove",
        "in a date palm grove", "in a mossy forest", "in a zen garden", "in a maze garden", "in a grass field",
        "in an alpine meadow", "on a rocky mountain ridge", "in a misty mountain valley", "near a glacial lake", 
        "in a dense tropical rainforest", "in an old-growth redwood forest", "in a misty cloud forest",
        "in a colorful autumn deciduous forest", "in a sparse boreal forest", "near a tranquil mountain stream",
        "in a red rock desert canyon", "among towering sand dunes", "in a rocky desert with cacti",
        "in a salt flat desert", "in a desert oasis with palm trees", "near a coral reef", "on a pebble beach",
        "on a rocky coastal cliff", "on a pristine white sand beach", "in a mangrove swamp",
        "in a rolling grassland prairie", "in an African savanna", "in a wildflower-filled meadow",
        "in a high-altitude steppe", "in a grassy wetland marsh", "near a melting glacier", "in a snowy taiga forest",
        "on the Arctic tundra", "near an Antarctic ice shelf", "in a field of Arctic wildflowers",
        "near an active volcano", "in a field of hardened lava", "near a bubbling mud pot",
        "next to a steaming geyser", "in a volcanic crater lake", "in a slot canyon", "among giant boulders",
        "on the banks of a meandering river", "near a thundering waterfall", "in a river canyon",
        "on a misty river at dawn", "near a series of cascading rapids", "on a snow-capped mountain peak",
        "among bizarre rock hoodoos", "in a limestone karst landscape", "near a natural stone arch",
        "on a remote tropical island", "on a volcanic island coastline", "in a lush island jungle interior",
        "on a windswept subarctic island", "near a fjord on a mountainous island"
        "in a verdant tea plantation with red-clothed pickers", "in a bamboo forest with golden sunlight filtering through",
        "in a lush rainforest with colorful tropical birds", "in a mossy ancient forest with pink cherry blossoms",
        "in a terraced rice field with workers in conical hats", "in a topiary garden with whimsical shapes",
        "in a dense fern gully with a small, clear stream", "in a tropical botanical garden with exotic flowers",
        "on a golf course with white sand bunkers", "in a vineyard with purple grapes ready for harvest",
        "in a field of tall grass with red poppies scattered throughout", "in a misty pine forest with orange mushrooms",
        "in an English garden maze with blooming roses", "in a lush valley with a rainbow arching overhead",
        "in a green tea field with Mount Fuji in the background", "in a traditional Peruvian weaving village",
        "in a vast sunflower field", "on a beach with golden sand and blue water", "in a field of yellow rapeseed flowers",
        "among fall foliage with golden leaves", "in a wheat field ready for harvest", "in a desert with golden sand dunes",
        "in a field of yellow tulips", "surrounded by autumn birch trees with yellow leaves", "in a field of yellow daffodils",
        "on a hillside covered in yellow wildflowers", "in a lemon grove with ripe yellow fruit", "in a field of goldenrod flowers",
        "among yellow aspen trees in autumn", "in a field of yellow marigolds", "surrounded by yellow ginkgo trees in fall"
        "in the village", "in the city", "at home", "on the couch", "in the livingroom", "near the fireplace",
        "in a cosy wooden cabin", "in a ski lodge", "in a mansion", "in a villa", "in a photography studio",
        "in a studio apartment", "in a penthouse apartment", "in Times Square, New York", "in the Red Square, Moscow",
        "in Los Angeles", "in Hollywood", "in a Bel Air villa", "in Paris", "in San Francisco", "in London", "in New York",
        "in Berlin", "in Tokyo", "in Chicago", "in Rome", "in Barcelona", "in Canada", "in Toronto", "in Alaska",
        "in Antarctica", "in the office", "at a luxury hotel", "in the kitchen", "at the balcony", "in a studio",
        "on the Great Wall of China", "in a historic castle", "at a famous landmark", "in an ancient ruin",
        "in a modern skyscraper", "on a bustling street market", "on a charming bridge", "at a picturesque harbor",
        "in a bustling cafe", "in a majestic palace", "in an art gallery", "in a world-famous museum", "in a theater",
        "in a fish market", "in a clock tower", "at a lighthouse", "in an old village", "in a professional photography studio",
        "at a historic monastery", "in an art deco building", "in a gothic cathedral", "at a scenic viewpoint",
        "at a picturesque quarry", "next to a windmill", "at a historic fort", "in an aquarium", "in a planetarium",
        "at a scenic dock", "on a historic ship", "in a bustling subway station", "in a busy city street", "on a yacht",
        "in a quiet village", "in a quiet library", "in a bustling airport terminal", "in a lively sports stadium",
        "in an elegant art gallery", "in a high-tech laboratory", "in an opulent palace", "in a cutting-edge skyscraper",
        "in a charming bed and breakfast", "in a bustling food market", "in a historic lighthouse",
        "in a whimsical fairy-tale inspired theme park", "at a remote Arctic research station", "on a cruise ship",
        "in a traditional Mongolian yurt camp", "at a bustling Broadway theater", "in a serene botanical garden",
        "next to a yellow stone wall", "next to a red brick wall", "next to a green tile wall", "next to a purple stone wall",
        "next to a blue stone wall", "next to a white stone wall", "next to a black stone wall", "next to a brown stone wall",
        "next to a wooden lime wall", "next to a orange brick wall", "next to a magenta stone wall", "next to a cyan stone wall",
        "next to a wooden wall", "next to a stone wall", "next to a marble wall", "next to a brick wall",
        "next to a glass window", "next to a yellow wall", "next to a red wall", "next to a green tile wall",
        "next to a purple wall", "next to a yellow wooden wall", "next to an orange stone wall",
        "next to a green stone wall", "next to a purple marble wall", "next to a blue marble wall",
        "next to a white marble wall", "next to a black marble wall", "next to a brown tile wall",
        "next to an ancient stone wall", "with a plain background", "with a blurred background",
        "against a white backdrop", "against a black backdrop", "with a colorful backdrop",
        "with a textured background", "with a gradient background", "with a bokeh effect background",
        "at the Grand Canyon", "next to Victoria Falls", "next to Niagara Falls", "at a music festival",
        "on a historic battlefield", "on a sailboat", "in a hot air balloon", "under the northern lights",
        "next to a historic statue", "in a traditional tea house", "in a serene butterfly sanctuary",
        "in an ancient underground cave system", "in a vibrant street art alley", "in a misty Scottish highland",
        "at a futuristic vertical farm", "in a traditional Japanese onsen", "at a bustling spice market in Marrakech",
        "in an otherworldly salt flat", "at a bioluminescent beach at night", "in a lush tropical treehouse resort",
        "at a historic Route 66 diner", "in a neon-lit cyberpunk cityscape", "in a tranquil lavender field in Provence",
        "in a serene Scandinavian fjord", "at a colorful hot air balloon festival",
        "in a mystical fog-covered ancient forest", "at a cutting-edge renewable energy farm",
        "outdoors", "indoors", "in an outdoor setting", "in a studio setting", "against an urban setting",
        "against a nature backdrop", "against a studio background", "against a beach scene"
        "in an old steel mill", "at a bustling shipyard", "in a modern automotive factory",
        "at an active construction site", "in a textile manufacturing plant", "at a wind turbine farm",
        "in a high-tech electronics assembly line", "at a busy seaport with cargo containers",
        "in a traditional blacksmith's workshop", "at a state-of-the-art recycling facility",
        "in a university lecture hall", "in a elementary school classroom", "at a public library reading room",
        "in a high school science lab", "at a coding bootcamp workspace", "in a music conservatory practice room",
        "at a culinary school kitchen", "in a medical school anatomy lab", "at an art school studio",
        "on a professional basketball court", "at an Olympic swimming pool", "in a state-of-the-art gymnasium",
        "on a golf course green", "at a baseball stadium dugout", "in a boxing ring corner", "on a soccer field sideline",
        "on an athletics track stadium", "at a professional football stadium", "on a track & field course",
        "on a tennis court baseline", "at a rock climbing wall", "in a yoga studio", "at a horse racing track",
        "in the cockpit of a commercial airliner", "on the deck of a luxury cruise ship",
        "at a bustling train station platform", "in the cabin of a high-speed bullet train",
        "at a busy airport terminal", "in the back of a yellow taxi cab", "on a city bus during rush hour",
        "in a sleek, modern subway car", "at a car rental facility", "in the control room of a cargo ship",
        "at a colorful Holi festival celebration", "during a traditional Japanese tea ceremony",
        "at a lively Carnival parade in Rio", "during a solemn Native American powwow",
        "at a vibrant Chinese New Year celebration", "during a formal Western wedding ceremony",
        "at a lively Oktoberfest beer hall", "during a traditional Indian Diwali festival",
        "at a Mexican Day of the Dead celebration", "during a Moroccan Ramadan evening feast",
        "in front of a large abstract mural", "surrounded by classical marble sculptures",
        "in a glass-blowing studio mid-creation", "at a pottery wheel shaping clay",
        "in front of a wall of colorful street art", "in a dance studio with mirrored walls",
        "at an outdoor installation art exhibit", "in a photography darkroom", "in a marine biology research vessel", 
        "at a bustling art gallery opening night", "in a theater prop and costume workshop",
        "in a cutting-edge robotics laboratory", "at a particle accelerator facility", "at an archaeological dig site",
        "in a clean room for semiconductor manufacturing", "at a radio telescope array", "in a renewable energy research center", 
        "in a genetic research laboratory", "at a weather monitoring station", "at a space mission control center",
    ]

    return random.choice(locations_settings_backgrounds_list)

def get_skin_description(ethnicity_group=None, stereotype_prob=0.7):
    skin_tones_by_ethnicity_dict = {
        "European":            ["fair", "light", "pale", "ivory", "porcelain", "rosy", "peach", "cream", "alabaster", "milky", "cool beige"],
        "Sub-Saharan African": ["dark brown", "deep brown", "chocolate", "ebony", "mahogany", "espresso", "rich brown"],
        "Middle Eastern":      ["olive", "tan", "medium", "golden", "warm beige", "light brown", "honey"],
        "Latin American":      ["olive", "tan", "caramel", "bronze", "golden", "medium", "coffee", "mocha"],
        "Oceanian":            ["tan", "golden brown", "deep brown", "bronze", "copper"],
        "Caribbean":           ["caramel", "golden brown", "deep brown", "mahogany", "cocoa"],
        "Central Asian":       ["light", "medium", "olive", "golden", "wheat"],
        "West Asian":          ["olive", "medium", "golden", "warm beige", "tan"],
        "North African":       ["olive", "tan", "golden", "caramel", "light brown", "medium brown"],
        "Scandinavian":        ["very fair", "pale", "porcelain", "ivory", "rosy"],
        "North American":      ["fair", "light", "medium", "olive", "tan", "brown", "dark brown"],
        "Arctic":              ["light", "fair", "golden", "ruddy"],
        "Southeast Asian":     ["light brown", "medium brown", "tan", "golden", "caramel"],
        "Balkan":              ["light", "medium", "olive", "golden", "tan"],
        "Polynesian":          ["golden brown", "tan", "bronze", "deep brown"],
        "Micronesian":         ["tan", "golden brown", "bronze", "medium brown"],
        "Melanesian":          ["deep brown", "dark brown", "ebony", "rich brown"],
        "Indigenous American": ["tan", "copper", "bronze", "reddish-brown", "golden brown"],
        "Australasian":        ["fair", "tan", "golden", "deep brown", "reddish-brown"],
        "Caucasian":           ["fair", "light", "pale", "ivory", "rosy", "peach", "beige"],
        "East Asian":          ["light", "fair", "ivory", "warm beige", "golden", "porcelain"],
        "South Asian":         ["tan", "caramel", "honey", "golden brown", "deep brown", "wheat", "bronze"]
    }

    skin_characteristics = [
        "smooth", "soft", "silky", "velvety", "radiant", "glowing", "dewy", "matte", "textured", "porous", 
        "freckled", "sun-kissed", "weathered", "leathery", "wrinkled", "lined", "age-spotted", "blemished", 
        "scarred", "pockmarked", "clear", "unblemished", "youthful", "mature", "supple", "firm", "taut", 
        "saggy", "dry", "oily", "combination", "sensitive", "rough", "calloused", "flushed", "ruddy", "pale", 
        "sallow", "ashen", "vibrant", "luminous", "dull", "mottled", "patchy", "even-toned", "uneven", 
        "translucent", "opaque", "porcelain-like", "alabaster-like", "bronzed", "sun-damaged", "tanned", 
        "untanned", "sunburnt", "detailed", "detailed texture", "detailed pores", "lustrous", "healthy", 
        "glassy", "plump", "hydrated", "moisturized", "flaky", "peeling", "bumpy", "dimpled", "velvety", 
        "buttery", "waxy", "papery", "tight", "loose", "elastic", "toned", "polished", "raw", "chapped"
    ]

    all_skin_tones = get_all_unique_dict_values(skin_tones_by_ethnicity_dict)
    if ethnicity_group is None or random.random() > stereotype_prob:
        tone = random.choice(all_skin_tones)
    else:
        tone = random.choice(skin_tones_by_ethnicity_dict.get(ethnicity_group, all_skin_tones))

    characteristic = random.choice(skin_characteristics)

    return f"{tone}, {characteristic} skin"

def get_hats_and_headwear(sex_group=None, stereotype_prob=0.8):
    hats_and_headwear_dict = {
        "Male": [
            "wearing a baseball cap", "with a fedora", "sporting a beanie", "with a flat cap", "wearing a turban",
            "wearing a cowboy hat", "with a top hat", "wearing a bowler hat", "with a newsboy cap",
            "sporting a trucker hat", "with a bucket hat", "wearing a military cap", "with a golf visor",
            "sporting a bandana", "with a beret", "wearing a sombrero", "with a turban", "with a mexican hat",
            "sporting a kippah", "with a fez", "wearing a ushanka", "with a porkpie hat", "with a scarf",
            "sporting a panama hat", "with a boater hat", "wearing a deerstalker", "with a trapper hat",
            "sporting a taqiyah", "with a tam o' shanter", "wearing a tricorn hat", "with a helmet", "with a steampunk top hat",
            "over-ear headphones", "with an earpice", "with in-ear headphones", "with a bluetooth headset", "with a VR headset",
        ],
        "Female": [
            "wearing a sun hat", "with a beret", "sporting a fascinator", "with a cloche hat", "with a traditional Chinese hair stick",
            "wearing a pillbox hat", "with a wide-brimmed hat", "with a headband", "with a headscarf",
            "wearing a beanie", "with a flower crown", "sporting a fedora", "with a bucket hat", "wearing a hijab",
            "wearing a turban", "with a baseball cap", "sporting a bandana", "with a hijab", "wearing a hair wrap",
            "wearing a veiled hat", "with a cowboy hat", "sporting a newsboy cap", "with a beret", "wearing a beaded African headwrap",
            "wearing a knit hat", "with a visor", "sporting a bonnet", "with a tam hat", "with a scarf",
            "wearing a cocktail hat", "with a lampshade hat", "sporting a toque", "with a furry trapper hat",
            "over-ear headphones", "with an earpice", "with in-ear headphones", "with a bluetooth headset", "with a VR headset",
        ],
        "Unisex": [
            "with a beanie", "wearing a snapback cap", "sporting a bucket hat", "with a bandana", "with a scrunchie", 
            "with a traditional Native American headdress", "with a military beret",
            "wearing a fedora", "with a baseball cap", "sporting a sun visor", "with a headband", "with hairpins",
            "wearing a flat cap", "with a beret", "sporting a trucker hat", "with a cowboy hat", "with a summer scarf",
            "wearing a knit cap", "with a military cap", "sporting a panama hat", "with a headscarf", "wearing a traditional Russian ushanka",
            "wearing a turban", "with a boonie hat", "sporting a cadet cap", "with a helmet", "with a winter scarf", "wearing a crown",
            "wearing a straw hat", "with a ski mask", "sporting a floppy hat", "with a trapper hat", "with decorative bobby pins",
        ]
    }
    
    all_headwear = get_all_unique_dict_values(hats_and_headwear_dict)
    if sex_group is None or random.random() > stereotype_prob:
        return random.choice(all_headwear)
    else:
        return random.choice(hats_and_headwear_dict.get(sex_group, all_headwear))

def get_random_jewelry(sex_group=None, stereotype_prob=0.8):
    jewelry_dict = {
        "Male": [
            "wearing a chain necklace", "with a simple ear stud", "sporting a bolo tie",
            "with a dog tag necklace", "wearing a small hoop earring", "with a nose stud",
            "sporting a tribal necklace", "with an eyebrow ring", "wearing a leather cord necklace",
            "with a septum piercing", "sporting a single diamond stud", "with a small gauge ear piercing",
            "wearing a thin gold chain", "with a curved barbell eyebrow piercing", "with a bowtie", 
            "with multiple ear piercings", "wearing a pendant necklace", "with a helix ear piercing",
            "with a tragus piercing", "wearing a silver chain", "sporting a shark tooth necklace",
            "with a crystal stud earring", "sporting a tongue piercing", "with a daith piercing"
        ],
        "Female": [
            "wearing hoop earrings", "with a pendant necklace", "sporting a choker", "with an eyebrow ring",
            "with pearl earrings", "wearing a statement necklace", "with chandelier earrings",
            "sporting a delicate chain necklace", "with a nose stud", "wearing drop earrings",
            "with a pearl choker", "sporting multiple ear piercings", "with a septum ring",
            "wearing a locket necklace", "with stud earrings", "sporting a bib necklace", "wearing geometric earrings",
            "with a nose ring", "wearing tassel earrings", "with a layered necklace", "wearing climbing vine ear cuffs",
            "sporting a crystal choker", "with ear cuffs", "wearing a cameo necklace",
            "with a tongue piercing", "sporting dangle earrings", "with a septum clicker", "with a bowtie",
        ],
        "Unisex": [
            "wearing a simple necklace", "with stud earrings", "sporting a nose ring", "with a bowtie",
            "with a choker", "wearing a pendant", "with multiple ear piercings", "with a dermal piercing",
            "sporting an ear cuff", "with a septum piercing", "wearing a chain necklace", 
            "with hoop earrings", "sporting a tongue stud", "with a cartilage piercing",
            "wearing a beaded necklace", "with a tragus piercing", "sporting a nose stud",
            "with an industrial bar piercing", "wearing a collar necklace", "with a conch piercing",
            "sporting a septum ring", "with dangle earrings", "wearing a torque necklace",
            "with a labret piercing", "sporting a helix piercing", "with a rook piercing", "with a lip ring", 
        ]
    }
    
    all_jewelry = get_all_unique_dict_values(jewelry_dict)
    if sex_group is None or random.random() > stereotype_prob:
        return random.choice(all_jewelry)
    else:
        return random.choice(jewelry_dict.get(sex_group, all_jewelry))

def get_weight_description():
    weight_descriptions_list = [
        "very slim", "slender", "lean", "willowy", "lithe", "waifish", "svelte",
        "thin", "skinny", "gaunt", "bony", "emaciated",
        "of average build", "with a moderate frame", "neither thin nor overweight",
        "with a balanced physique", "of normal weight", "with a typical body type",
        "curvy", "full-figured", "plump", "chubby", "rounded", "soft",
        "with a bit of extra weight", "slightly heavy-set",
        "heavyset", "portly", "stout", "corpulent", "rotund", "plush",
        "plus-sized", "full-bodied", "generously proportioned",
        "muscular", "athletic", "well-built", "toned", "fit", "strapping",
        "brawny", "burly", "robust", "solid",
        "with a unique body type", "with a distinctive physique",
        "with an unconventional build", "with an interesting silhouette"
    ]
    return random.choice(weight_descriptions_list)

def get_random_time_of_day():
    times_of_day_list = [
        'at dawn', 'at dusk', 'at twilight', 'during sunset', 'during sunrise', 'at midnight', 'at afternoon', 
        'at late afternoon', 'at golden hour', 'at midday', 'at noon', 'at night', 'in the evening', 'in the morning', 
        'in the afternoon', 'at the stroke of midnight', 'in the wee hours', 'at the crack of dawn', 'at high noon',
        'during the witching hour', 'at brunch time', 'during tea time', 'at supper time', 'during happy hour',
        'at the eleventh hour', 'during siesta time', 'at bedtime', 'at the break of day', 'during the dog days of summer',
        'during early morning', 'during late morning', 'during early evening', 'during late evening',
        'at cocktail hour', 'during lunchtime', 'during dinnertime', 'at the blue hour', 'at the magic hour',
        'during rush hour', 'at daybreak', 'at sundown', 'during civil twilight', 'during nautical twilight',
        'during astronomical twilight', 'at first light', 'at last light', 'during solar noon', 'during solar midnight'
    ]
    time_of_day = np.random.choice(times_of_day_list)
    return time_of_day

def get_random_weather_condition():
    weather_conditions_list = [
        'while its raining', 'while its snowing', 'when scorching hot', 'in perfect weather',
        'during a thunderstorm', 'during a heatwave', 'during a cold snap', 'during a drizzle', 'during a hailstorm',
        'during a sandstorm', 'during a snowstorm', 'during a windstorm', 'during a foggy day', 'during a cloudy day',
        'during a sunny day', 'during an overcast day', 'during a monsoon', 'during a hurricane', 'during a tornado',
        'during a blizzard', 'during an earthquake', 'during a solar eclipse', 'during a lunar eclipse', 'during a meteor shower',
        'during high tide', 'during low tide', 'during a rainbow', 'during a flood', 'during a drought',
        'during a wildfire', 'during a volcanic eruption', 'during an avalanche', 'during a cyclone', 'during a typhoon',
        'during an ice storm', 'during a misty morning', 'during a humid afternoon', 'during a dry evening', 'during a muggy night'
    ]
    weather_condition = np.random.choice(weather_conditions_list)
    return weather_condition

def get_eye_description(ethnicity_group=None, stereotype_prob=0.3):

    all_eye_colors = get_all_unique_dict_values(eye_colors_dict)
    if ethnicity_group is None or np.random.rand() > stereotype_prob:
        eye_colors_list = all_eye_colors
    else:
        eye_colors_list = eye_colors_dict.get(ethnicity_group, all_eye_colors)

    color = random.choice(eye_colors_list)

    eye_styles_list = [
        "Captivating {color} eyes", "{color} eyes lost in thought", "Piercing {color} eyes", 'huge {color} eyes',
        "Striking {color} eyes", "{color} eyes, large and expressive", "Small {color} eyes", 'large {color} eyes',
        "Inviting {color} eyes", "{color} eyes, wide open", "Closed, {color} eyes", "Sparkling {color} eyes",
        "Twinkling {color} eyes", "plain {color} eyes", "Regular {color} eyes", "Mysterious {color} eyes",
        "Expressive {color} eyes", "Glistening {color} eyes", "Luminous {color} eyes", "Focused {color} eyes",
        "Dreamy {color} eyes", "intense {color} eyes", "Gentle {color} eyes", "Curious {color} eyes", 'large striking {color} eyes',
        "Alluring {color} eyes", "haunting {color} eyes", "Innocent {color} eyes", "Hypnotic {color} eyes",
        "Mesmerizing {color} eyes", "animated {color} eyes", "Sleepy {color} eyes", "Observant {color} eyes",
        "deep set {color} eyes", "bulging {color} eyes", "Hooded, {color} eyes", "Almond shaped, {color} eyes",
        "round {color} eyes", "wide {color} eyes", "Narrow, {color} eyes", "Cat-like {color} eyes",
        "winking {color} eyes", "dilated pupil, {color} eyes", "{color} eyes with an enigmatic hue", 'deep {color} eyes',
        "{color} eyes radiating vibrant light", "{color} eyes deep in introspection", "{color} eyes resolute and firm",
        "{color} eyes reflecting a whimsical sparkle", "{color} eyes sorrowful and deep", 
        "{color} eyes brimming with joy", "{color} eyes in a contemplative state", 
        "{color} eyes emanating serenity", "{color} eyes ablaze with intensity"
    ]

    eye_style = random.choice(eye_styles_list)
    eye_description = eye_style.format(color=color)

    return eye_description

def get_clothing_description(sex_group=None, sterotype_prob=0.6):

    clothing_color_list = random.choice([
        "red", "blue", "green", "yellow", "purple", "pink", "orange",
        "black", "white", "gray", "brown", "navy", "teal", "maroon", "olive",
        "beige", "turquoise", "lavender", "crimson", "indigo", "magenta",
        "chartreuse", "burgundy", "periwinkle", "coral", "mustard", "plum",
        "khaki", "mauve", "salmon", "mint", "gold", "silver", "bronze",
        "copper", "platinum", "pastel pink", "pastel blue", "pastel green",
        "pastel yellow", "pastel purple", "emerald", "sapphire", "ruby",
        "amethyst", "topaz", "garnet", "ivory", "cream", "tan", "taupe",
        "charcoal", "slate"
    ])

    clothing_patterns_list = [
        "striped", "polka dot", "floral", "plaid", "checkered", "paisley",
        "herringbone", "houndstooth", "geometric pattern"
    ]

    clothing_types_dict = {
        "casual": {
            "neutral": ["t-shirt", "jeans", "sweater", "hoodie", "shorts", "tracksuit"],
            "male": ["polo shirt"],
            "female": ["leggings", "yoga pants", "tank top"]
        },
        "formal": {
            "neutral": ["suit"],
            "male": ["tuxedo", "dress shirt", "tie"],
            "female": ["cocktail dress", "evening gown", "blouse", "pencil skirt"]
        },
        "professional": {
            "neutral": ["business suit", "slacks"],
            "male": ["necktie"],
            "female": ["pantsuit", "blouse", "knee-length skirt"]
        },
        "outerwear": {
            "neutral": ["jacket", "coat", "trench coat", "parka", "windbreaker", "peacoat", "leather jacket", "denim jacket", "bomber jacket"]
        },
        "dresses_skirts": {
            "female": ["sundress", "maxi dress", "mini skirt", "midi skirt", "wrap dress", "shirt dress", "A-line dress", "pleated skirt", "tulle skirt"],
            "male": ["scottish kilt", "togas"]
        },
        "ethnic": {
            "neutral": ["kaftan", "poncho", "tunic"],
            "male": ["kurta", "sherwani", "kilt", "lederhosen"],
            "female": ["sari", "cheongsam", "dirndl", "ao dai", "hanbok", "qipao", "yukata"]
        },
        "uniform": {
            "neutral": ["military uniform", "police uniform", "firefighter uniform", "doctor's white coat", "chef's uniform", "pilot's uniform", "nurse's scrubs", "judge's robe", "academic regalia"]
        },
        "sports": {
            "neutral": ["soccer jersey", "basketball uniform", "tennis whites", "cycling gear", "martial arts gi"],
            "female": ["leotard", "gymnast outfit", "yoga attire"]
        },
        "workwear": {
            "neutral": ["overalls", "coveralls", "high-visibility vest", "lab coat", "welder's protective gear"]
        },
        "unique": {
            "neutral": ["avant-garde designer piece", "futuristic bodysuit", "steampunk-inspired outfit", "cyberpunk ensemble"]
        },
        "historical": {
            "neutral": ["Renaissance costume"],
            "male": ["Victorian-era suit", "1920s gangster style"],
            "female": ["Victorian-era dress", "1920s flapper style", "1950s rockabilly fashion"]
        },
        "religious": {
            "neutral": ["ceremonial tribal wear", "traditional wedding attire"],
            "male": ["monk's robe"],
            "female": ["nun's habit"]
        }
    }

    if sex_group is None or np.random.rand() > sterotype_prob:
        sex_group = random.choice(['male', 'female', 'neutral'])

    assert sex_group in ['male', 'female', 'neutral']

    clothing_category = random.choice(list(clothing_types_dict.keys()))
    all_clothing_category_items = get_all_unique_dict_values(clothing_types_dict[clothing_category])
    clothing_items_list = clothing_types_dict[clothing_category].get(sex_group, all_clothing_category_items)
    if sex_group in ['male', 'female']:
        clothing_items_list += clothing_types_dict[clothing_category].get('neutral', all_clothing_category_items)
    
    clothing_item = random.choice(clothing_items_list)
    clothing_color = random.choice(clothing_color_list)

    if clothing_category in ["casual", "formal", "professional", "outerwear", "dresses_skirts"]:
        if random.random() < 0.2:
            pattern = random.choice(clothing_patterns_list)
            clothing_str = f"wearing a {pattern} {clothing_item}"
        else:
            clothing_str = f"wearing a {clothing_color} {clothing_item}"
    elif clothing_category in ["ethnic", "unique", "historical"]:
        clothing_str = f"dressed in {clothing_item}"
    elif clothing_category in ["uniform", "sports", "workwear", "religious"]:
        clothing_str = f"in {clothing_item}"
    else:
        clothing_str = f"wearing {clothing_item}"

    # if we are not using a stereotype, we can mix and match clothing items from different categories
    if np.random.rand() > sterotype_prob:
        clothing_items_list = [item for sublist in clothing_types_dict[clothing_category].values() for item in sublist]
        clothing_str = f"wearing {random.choice(clothing_items_list)}"
        return clothing_str

    return clothing_str

def get_random_modifier_string():

    modifier_str = ''
    if np.random.rand() < 0.2:
        modifier_str = modifier_str + 'wearing traditional attire, '
    if np.random.rand() < 0.1:
        modifier_str = modifier_str + 'casual pose, '
    if np.random.rand() < 0.3:
        modifier_str = modifier_str + np.random.choice(['photography, ', 'professional photography, ', 'photorealism, ', 'ultrarealistic uhd faces, '])
    if np.random.rand() < 0.2:
        modifier_str = modifier_str + np.random.choice(['hyper realism, ',  'realistic, ', 'ultra realistic, ',
                                                        'highly detailed, ', 'very detailed, ', 'hyper detailed, ', 'detailed, '])
    if np.random.rand() < 0.6:
        modifier_str = modifier_str + np.random.choice(['detailed skin, ', 'detailed skin texture, ', 'detailed skin pores, '])
    if np.random.rand() < 0.5:
        modifier_str = modifier_str + 'bokeh, '
    if np.random.rand() < 0.5:
        modifier_str = modifier_str + np.random.choice(['film, ', 'still from a film, ', 'raw candid cinema, ', 'cinematic movie still, '])
    if np.random.rand() < 0.2:
        modifier_str = modifier_str + np.random.choice(['head shot, ', 'medium shot, ', 'wide shot, ', 'zoomed out, '])
    if np.random.rand() < 0.2:
        modifier_str = modifier_str + np.random.choice(['triadic color scheme, ', 'vivid color, ', 'remarkable color, ', 'color graded, '])
    if np.random.rand() < 0.5:
        modifier_str = modifier_str + np.random.choice(['studio lighting, ', 'volumetric lighting, ', 'subsurface scatter, ', 'natural light, ', 'soft light, ', 'hard light, ',
                                                        'atmospheric lighting, ', 'cinematic lighting, ', 'dramatic lighting, ', 'hard rim lighting photography, '])
    if np.random.rand() < 0.3:
        modifier_str = modifier_str + np.random.choice(['4k, ', '8k, ', 'uhd, ', 'ultra hd, ', 'high quality, ', 'HDR, '])
    if np.random.rand() < 0.3:
        modifier_str = modifier_str + np.random.choice(['nikon d850, ', 'kodachrome 25, ', 'kodak ultra max 800, ', 'kodak portra 160, ', 'DSLR camera, ',
                                                        'canon eos r3, ', 'Ilford HP5 400, ', 'samsung nx300m, ', 'sony a6000, ', 'olympus om-d, ',
                                                        'panasonic lumix dmc-gx85, ', 'fujifilm x70, ', 'canon eos, ', 'color graded porta 400 film, '])
    if np.random.rand() < 0.1:
        modifier_str = modifier_str + np.random.choice(['120mm, ', '85mm, ', '50mm, ', '35mm, '])
    if np.random.rand() < 0.1:
        modifier_str = modifier_str + np.random.choice(['f/1.4, ', 'f/2.5, ', 'f/3.2, ', 'f/1.1, '])
    if np.random.rand() < 0.1:
        modifier_str = modifier_str + np.random.choice(['35mm film roll photo, ', 'film, ', 'porta 400 film, '])
    if np.random.rand() < 0.1:
        modifier_str = modifier_str + np.random.choice(['iso 120, ', 'iso 210, '])
    if np.random.rand() < 0.1:
        modifier_str = modifier_str + np.random.choice(['wide lens, ', 'lens flare, ', 'sharp focus, ', 'hasselblad, '])
    if np.random.rand() < 0.3:
        modifier_str = modifier_str + np.random.choice(['alluring, ', 'beautiful, ', 'breath-taking, ', 'captivating, ', 'addorable, ', 'intricate, ',
                                                        'chic, ', 'classy, ', 'curvaceous, ', 'breath-cute, ', 'fashionable, ', 'elegant, ',
                                                        'gorgeous, ', 'graceful, ', 'lovely, ', 'mesmerizing, ', 'petite, ', 'pretty, ', 'tall, ',
                                                        'radiant, ', 'ravishing, ', 'slim, ', 'stunning, ', 'stylish, ', 'sultry, ', 'sweet, ',
                                                        'affectionate, ', 'ardent, ', 'articulate, ', 'at ease, ', 'attentive, ', 'awake, ',
                                                        'aware, ', 'boyish, ', 'brave, ', 'broad-shouldered, ', 'calm, ', 'voluptuous, ',
                                                        'caring, ', 'centered, ', 'charming, ', 'chiseled cheekbones, ', 'sharp features, ',
                                                        'classic good looks, ', 'clean-shaven, ', 'clever, ', 'compassionate, ', 'candid, ', 'attractive, ',
                                                        'confident, ', 'conscious, ', 'considerate, ', 'content, ', 'cosmopolitan, ',
                                                        'courageous, ', 'courteous, ', 'cultured, ', 'dark skin, ', 'dashing, ',
                                                        'debonair, ', 'defined jawline, ', 'devoted, ', 'educated, ', 'eloquent, ',
                                                        'faithful, ', 'fearless, ', 'firm skin, ', 'focused, ', 'full lips, ',
                                                        'fully engaged, ', 'gentle, ', 'glowing skin, ', 'grounded, ', 'handsome, ',
                                                        'handsome features, ', 'in the moment, ', 'insightful, ', 'intelligent, ',
                                                        'intense, ', 'kind, ', 'loyal, ', 'mannerly, ', 'mischievous, ', 'muscular, ',
                                                        'olive skin, ', 'passionate, ', 'peaceful, ', 'polite, ', 'porcelain skin, ',
                                                        'present, ', 'refined, ', 'reliable, ', 'rugged, ', 'secure, ', 'self assured, ',
                                                        'sensitive, ', 'serene, ', 'smooth skin, ', 'soft skin, ', 'sophisticated, ',
                                                        'square-jawed, ', 'stable, ', 'strong, ', 'strong chin, ', 'suave, ',
                                                        'sun kissed skin, ', 'thick, ', 'trustworthy, ', 'twinkling eyes, ', 'urbane, ',
                                                        'well mannered, ', 'well spoken, ', 'well traveled, ', 'well-built, ', 'witty, ', 'worldly, '])
    if np.random.rand() < 0.2:
        modifier_str = modifier_str + np.random.choice(['subtle shadows, ', 'shadow, '])
    if np.random.rand() < 0.05:
        modifier_str = modifier_str + np.random.choice(['mist, ', 'wet, ', 'foggy background, '])
    if np.random.rand() < 0.05:
        modifier_str = modifier_str + 'golden ratio composition, '
    if np.random.rand() < 0.05:
        modifier_str = modifier_str + 'dramatic, '
    if np.random.rand() < 0.05:
        modifier_str = modifier_str + 'award winning photograph, '
    if np.random.rand() < 0.05:
        modifier_str = modifier_str + 'epic composition, '
    if np.random.rand() < 0.05:
        modifier_str = modifier_str + 'pexels, '
    if np.random.rand() < 0.05:
        modifier_str = modifier_str + 'high contrast, '

    return modifier_str[:-2]

def get_all_unique_dict_values(property_dict):
    all_unique_values = []
    for key, values in property_dict.items():
        all_unique_values.extend(values)
    all_unique_values = list(set(all_unique_values))

    return all_unique_values

def get_random_ethnicity(ethnicity_group=None, top_level_prob=0.5):    
    if ethnicity_group is None:
        ethnicity_group = np.random.choice(list(ethnicities_dict.keys()))

    if np.random.rand() < top_level_prob:
        ethnicity = ethnicity_group
    else:
        ethnicity = np.random.choice(ethnicities_dict[ethnicity_group])
    
    return ethnicity

def get_age_sex_ethnicity(ethnicity_group=None, sex_group=None, age_group=None):
    # Age group definitions
    age_groups = {
        "baby": (1, 18),          # 1-18 months
        "toddler": (1, 4),        # 1-4 years
        "child": (4, 12),         # 4-12 years
        "teenager": (13, 19),     # 13-19 years
        "young adult": (18, 38),  # 18-38 years
        "middle-aged": (30, 55),  # 30-55 years
        "elderly": (50, 100)      # 50-100 years
    }

    # sample age group if not provided
    if age_group is None:
        age_group = np.random.choice(list(age_groups.keys()))

    # sample sex group if not provided
    if sex_group is None:
        sex_group = np.random.choice(['male', 'female'])

    # sample ethnicity based on the cluster
    ethnicity = get_random_ethnicity(ethnicity_group)

    # sample age based on age group range
    age_min, age_max = age_groups[age_group]
    age = np.random.choice(range(age_min, age_max + 1))

    if age_group == 'baby':
        sex = 'boy' if sex_group == 'male' else 'girl'
        age_sex_ethnicity_str = f'{age} month old {ethnicity} baby {sex}'
    elif age_group == 'toddler':
        sex = 'boy' if sex_group == 'male' else 'girl'
        age_sex_ethnicity_str = f'{age} year old {ethnicity} toddler {sex}'
    elif age_group == 'child':
        sex = 'boy' if sex_group == 'male' else 'girl'
        age_sex_ethnicity_str = f'{age} year old {ethnicity} {sex}'
    elif age_group == 'teenager':
        sex = 'boy' if sex_group == 'male' else 'girl'
        age_sex_ethnicity_str = f'{age} year old {ethnicity} teenage {sex}'
    elif age_group == 'young adult':
        if sex_group == 'male':
            sex = np.random.choice(['man', 'guy', 'person', 'male', 'brother'])
        else:
            sex = np.random.choice(['woman', 'dame', 'lady', 'female', 'sister'])
        sex = f'young {sex}' if np.random.rand() < 0.25 else sex
        age_sex_ethnicity_str = f'{age} year old {ethnicity} {sex}'
    elif age_group == 'middle-aged':
        if sex_group == 'male':
            sex = np.random.choice(['man', 'guy', 'husband', 'person', 'male', 'father', 'gentleman'])
        else:
            sex = np.random.choice(['woman', 'dame', 'wife', 'lady', 'female', 'mother', 'gentlewoman'])
        sex = f'middle-aged {sex}' if np.random.rand() < 0.25 else sex
        age_sex_ethnicity_str = f'{age} year old {ethnicity} {sex}'
    elif age_group == 'elderly':
        if sex_group == 'male':
            sex = np.random.choice(['man', 'grandfather', 'grandpa', 'person', 'father', 'husband', 'gentleman'])
        else:
            sex = np.random.choice(['woman', 'grandmother', 'grandma', 'lady', 'mother', 'wife', 'gentlewoman'])
        sex = f'elderly {sex}' if np.random.rand() < 0.25 else sex
        age_sex_ethnicity_str = f'{age} year old {ethnicity} {sex}'

    return age_sex_ethnicity_str

def get_random_expression():
    return random.choice(expressions_list)

def get_lighting_atmosphere(lighting_category=None):
    if lighting_category is None:
        lighting_category = random.choice(list(lighting_descriptions_dict.keys()))
    return random.choice(lighting_descriptions_dict[lighting_category])

def get_hair_description(ethnicity_group=None, sterotype_prob=0.3):

    all_hair_colors = get_all_unique_dict_values(hair_colors_dict)
    if ethnicity_group is None or np.random.rand() > sterotype_prob:
        hair_colors_list = all_hair_colors
    else:
        hair_colors_list = hair_colors_dict.get(ethnicity_group, all_hair_colors)
    
    color = random.choice(hair_colors_list)
    style = random.choice(hair_styles_list)
    return f"with {color} {style}"

def get_random_face_pose():
    return random.choice(face_poses_list)


#%% main face prompt generator function

def generate_face_prompt(ethnicity_group=None, sex_group=None, age_group=None,
                         lighting_category=None, num_elements_to_add=None):
    
    # Sample demographic information
    if ethnicity_group is None:
        ethnicity_group = random.choice(list(ethnicities_dict.keys()))
    if sex_group is None:
        sex_group = np.random.choice(['male', 'female'], p=[0.5, 0.5])
    if age_group is None:
        age_group = np.random.choice(['baby', 'toddler', 'child', 'teenager', 'young adult', 'middle-aged', 'elderly'],
                                     p=[0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.25])

    # Generate base prompt
    prompt_start = get_prompt_start()
    age_sex_ethnicity = get_age_sex_ethnicity(ethnicity_group, sex_group, age_group)
    base_prompt = f"{prompt_start} {age_sex_ethnicity}, "

    # Generate additional elements
    elements = [
        get_random_face_pose(),
        get_random_gaze_direction(),
        get_hair_description(ethnicity_group),
        get_eye_description(ethnicity_group),
        get_skin_description(ethnicity_group),
        get_clothing_description(sex_group),
        get_hats_and_headwear(sex_group),
        get_random_jewelry(sex_group),
        get_random_glasses(),
        get_random_time_of_day(),
        get_random_weather_condition(),
        get_weight_description(),
        get_random_modifier_string(),
        get_lighting_atmosphere(lighting_category),
    ]
    elements = elements + elements[-3:]  # repeat last 3 elements to increase their probability
    
    # add facial hair possibility when needed
    if sex_group == 'male' and age_group in ['teenager', 'young adult', 'middle-aged', 'elderly']:
        elements.append(ger_facial_hair_description())

    # add makeup possibility when needed
    if sex_group == 'female' and age_group in ['teenager', 'young adult', 'middle-aged', 'elderly']:
        elements.append(get_makeup_description())
        elements = elements + [elements[-1]] # repeat makeup description to increase its probability
        elements = elements + [elements[-1]] # repeat makeup description to increase its probability

    # Randomly select subset of elements
    if num_elements_to_add is None:
        num_elements_to_add = random.randint(4, 9)
    selected_elements = random.sample(elements, min(num_elements_to_add, len(elements)))
    selected_elements = list(set(selected_elements)) # remove duplicates
    selected_elements = [get_random_expression()] + selected_elements # always add expression at the beginning
    selected_elements.append(get_location_setting_background()) # always add location setting background at the end

    # Combine base prompt with selected elements
    full_prompt = f"{base_prompt} {', '.join(selected_elements)}"

    return full_prompt

def display_conditions(conditions_dict):
    print('Conditions:')
    print('-----------')
    for key, value in conditions_dict.items():
        print(f'  {key} = {value}')

def get_formatted_prompt_for_display(prompt, max_line_length=85):
    parts = [part.strip() for part in prompt.split(',')]
    formatted_prompt = ""
    current_line = ""

    for i, part in enumerate(parts):
        if len(current_line) + len(part) > max_line_length:
            if formatted_prompt:
                formatted_prompt += ',\n'
            formatted_prompt += current_line
            current_line = part
        else:
            if current_line:
                current_line += ", " + part
            else:
                current_line = part

    if current_line:
        if formatted_prompt:
            formatted_prompt += ',\n'
        formatted_prompt += current_line

    return formatted_prompt


#%% main test

if __name__ == "__main__":
    num_samples_per_type = 10
    show_conditional_sampling = True
    show_conditional_sampling = False
    print("Generating face prompts...\n")

    all_ethinicity_groups = list(ethnicities_dict.keys())
    all_lighting_categories = list(lighting_descriptions_dict.keys())
    all_age_groups = ['baby', 'toddler', 'child', 'teenager', 'young adult', 'middle-aged', 'elderly']
    all_sex_groups = ['male', 'female']

    print('=' * 100)
    print("1. Unconditioned sampling:")
    for i in range(num_samples_per_type):
        prompt = generate_face_prompt()
        print(f"Prompt {i+1}: \n----------")
        print(get_formatted_prompt_for_display(prompt))
        print()
    print('=' * 100)
    print('\n\n')

    if show_conditional_sampling:
        print('=' * 100)
        print("2. Partially Conditioned sampling:")
        conditions_dict_list = [
            {"ethnicity_group": random.choice(all_ethinicity_groups), 'sex_group': random.choice(all_sex_groups), 'age_group': random.choice(all_age_groups)},
            {"ethnicity_group": random.choice(all_ethinicity_groups), 'sex_group': random.choice(all_sex_groups)},
            {'age_group': random.choice(all_age_groups), "lighting_category": random.choice(all_lighting_categories)},
            {'sex_group': random.choice(all_sex_groups), 'age_group': random.choice(all_age_groups)},
            {'sex_group': random.choice(all_sex_groups), 'age_group': random.choice(all_age_groups)},
            {"lighting_category": random.choice(all_lighting_categories), "num_elements_to_add": np.random.randint(2, 5)},
            {"lighting_category": random.choice(all_lighting_categories), "num_elements_to_add": np.random.randint(4, 10)},
            {"lighting_category": random.choice(all_lighting_categories), "num_elements_to_add": np.random.randint(7, 12)},
        ]

        for i, conditions_dict in enumerate(conditions_dict_list):
            prompt = generate_face_prompt(**conditions_dict)
            print('=' * 80)
            display_conditions(conditions_dict)
            print('-' * 40)
            print(f"Prompt {i+1}: \n---------")
            print(get_formatted_prompt_for_display(prompt))
            print('=' * 80)
        print('=' * 100)
        print('\n\n')

        print('=' * 100)
        print("3. Fully Conditioned sampling:")
        for i in range(num_samples_per_type):
            ethnicity_group = random.choice(all_ethinicity_groups)
            sex_group = random.choice(all_sex_groups)
            age_group = random.choice(all_age_groups)
            lighting_category = random.choice(all_lighting_categories)
            num_elements_to_add = np.random.randint(1, 5)

            prompt = generate_face_prompt(
                ethnicity_group=ethnicity_group, 
                sex_group=sex_group,
                age_group=age_group,
                lighting_category=lighting_category,
                num_elements_to_add=num_elements_to_add
            )

            conditions_dict = {
                'ethnicity_group': ethnicity_group, 
                'sex_group': sex_group,
                'age_group': age_group,
                'lighting_category': lighting_category,
                'num_elements_to_add': num_elements_to_add
            }

            print('=' * 80)
            display_conditions(conditions_dict)
            print('-' * 40)
            print(f"Prompt {i+1}: \n---------")
            print(get_formatted_prompt_for_display(prompt))
            print('=' * 80)
        print('=' * 100)
        print('\n\n')

# %%
