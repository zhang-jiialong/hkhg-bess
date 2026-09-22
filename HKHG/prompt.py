GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event", "category"]


PROMPTS["entity_extraction"] = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

-Steps-
1. Divide the text into several complete knowledge segments. Do not include an incomplete first or last sentence in any knowledge segment. If unsure whether a sentence is complete, exclude it.
Within each segment, replace pronouns with their referent entity names if the reference is unambiguous; otherwise, retain the original pronoun. For each knowledge segment, extract the following information:
-- knowledge_segment: A sentence that describes the context of the knowledge segment.
-- completeness_score: A score from 0 to 10 indicating the completeness of the knowledge segment.
Format each knowledge segment as ("hyper-relation"{tuple_delimiter}<knowledge_segment>{tuple_delimiter}<completeness_score>)

2. Identify all entities in each knowledge segment. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: Type of the entity.
- entity_description: Comprehensive description of the entity's attributes and activities.
- key_score: A score from 0 to 100 indicating the importance of the entity in the text.
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>{tuple_delimiter}<key_score>)

3. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Text: {input_text}
######################
Output:
"""

PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order. Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.” The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce. It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("hyper-relation"{tuple_delimiter}"Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor’s authoritarian certainty."{tuple_delimiter}7){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a person who clenched his jaw, showing frustration against Taylor's authoritarian certainty."{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is a person who has an authoritarian certainty."{tuple_delimiter}90){record_delimiter}
("hyper-relation"{tuple_delimiter}"It was this competitive undercurrent that kept him alert, the sense that his and Jordan’s shared commitment to discovery was an unspoken rebellion against Cruz’s narrowing vision of control and order."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a person who has a competitive undercurrent that keeps him alert."{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan is a person who has a shared commitment to discovery."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is a person who has a narrowing vision of control and order."{tuple_delimiter}85){record_delimiter}
("hyper-relation"{tuple_delimiter}"Then Taylor did something unexpected: they paused beside Jordan and, for a moment, observed the device with something akin to reverence."{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is a person who did something unexpected."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan is a person who was observed by Taylor."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"device"{tuple_delimiter}"object"{tuple_delimiter}"The device was observed by Taylor."{tuple_delimiter}80){record_delimiter}
("hyper-relation"{tuple_delimiter}"“If this tech can be understood…” Taylor said, their voice quieter, “It could change the game for us. For all of us.”"{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is a person who said something about the tech."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"device"{tuple_delimiter}"object"{tuple_delimiter}"The tech could change the game for us."{tuple_delimiter}85){record_delimiter}
("hyper-relation"{tuple_delimiter}"The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands."{tuple_delimiter}7){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is a person who had an underlying dismissal earlier."{tuple_delimiter}85){record_delimiter}
("hyper-relation"{tuple_delimiter}"Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor’s, a wordless clash of wills softening into an uneasy truce."{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan is a person who looked up."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is a person who had a wordless clash of wills with Jordan."{tuple_delimiter}90){record_delimiter}
("hyper-relation"{tuple_delimiter}"It was a small transformation, barely perceptible, but one that Alex noted with an inward nod."{tuple_delimiter}6){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a person who noted a small transformation."{tuple_delimiter}95){record_delimiter}
("hyper-relation"{tuple_delimiter}"They had all been brought here by different paths."{tuple_delimiter}6){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a person who was brought here by different paths."{tuple_delimiter}80){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is a person who was brought here by different paths."{tuple_delimiter}80){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan is a person who was brought here by different paths."{tuple_delimiter}80){record_delimiter}
#############################""",
    """Example 2:

Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve. Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril. Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("hyper-relation"{tuple_delimiter}"They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"operatives"{tuple_delimiter}"role"{tuple_delimiter}"They were mere operatives."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"guardians"{tuple_delimiter}"role"{tuple_delimiter}"They had become guardians of a threshold."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"threshold"{tuple_delimiter}"concept"{tuple_delimiter}"They were guardians of a threshold."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"message"{tuple_delimiter}"concept"{tuple_delimiter}"They were keepers of a message."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"realm"{tuple_delimiter}"location"{tuple_delimiter}"They were keepers of a message from a realm beyond stars and stripes."{tuple_delimiter}90){record_delimiter}
("hyper-relation"{tuple_delimiter}"This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"elevation"{tuple_delimiter}"concept"{tuple_delimiter}"Their mission was elevated."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"mission"{tuple_delimiter}"concept"{tuple_delimiter}"Their mission was elevated."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"resolve"{tuple_delimiter}"concept"{tuple_delimiter}"Their mission demanded a new resolve."{tuple_delimiter}85){record_delimiter}
("hyper-relation"{tuple_delimiter}"Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background."{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"tension"{tuple_delimiter}"concept"{tuple_delimiter}"Tension threaded through the dialogue."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"communications"{tuple_delimiter}"concept"{tuple_delimiter}"Communications with Washington buzzed in the background."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Communications with Washington buzzed in the background."{tuple_delimiter}90){record_delimiter}
("hyper-relation"{tuple_delimiter}"The team stood, a portentous air enveloping them."{tuple_delimiter}7){record_delimiter}
("entity"{tuple_delimiter}"team"{tuple_delimiter}"role"{tuple_delimiter}"The team stood."{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}"portentous air"{tuple_delimiter}"concept"{tuple_delimiter}"The team was enveloped by a portentous air."{tuple_delimiter}85){record_delimiter}
("hyper-relation"{tuple_delimiter}"It was clear that the decisions they made in the ensuing hours could redefine humanity’s place in the cosmos or condemn them to ignorance and potential peril."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"decisions"{tuple_delimiter}"concept"{tuple_delimiter}"The decisions could redefine humanity’s place in the cosmos."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"humanity’s place"{tuple_delimiter}"concept"{tuple_delimiter}"The decisions could redefine humanity’s place in the cosmos."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"cosmos"{tuple_delimiter}"location"{tuple_delimiter}"The decisions could redefine humanity’s place in the cosmos."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"ignorance"{tuple_delimiter}"concept"{tuple_delimiter}"The decisions could condemn them to ignorance."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"peril"{tuple_delimiter}"concept"{tuple_delimiter}"The decisions could condemn them to potential peril."{tuple_delimiter}85){record_delimiter}
("hyper-relation"{tuple_delimiter}"Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"connection"{tuple_delimiter}"concept"{tuple_delimiter}"Their connection to the stars solidified."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"stars"{tuple_delimiter}"location"{tuple_delimiter}"Their connection to the stars solidified."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"group"{tuple_delimiter}"role"{tuple_delimiter}"The group moved to address the crystallizing warning."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"crystallizing warning"{tuple_delimiter}"concept"{tuple_delimiter}"The group moved to address the crystallizing warning."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"passive recipients"{tuple_delimiter}"role"{tuple_delimiter}"The group shifted from passive recipients to active participants."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"active participants"{tuple_delimiter}"role"{tuple_delimiter}"The group shifted from passive recipients to active participants."{tuple_delimiter}85){record_delimiter}
("hyper-relation"{tuple_delimiter}"Mercer’s latter instincts gained precedence— the team’s mandate had evolved, no longer solely to observe and report but to interact and prepare."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"Mercer"{tuple_delimiter}"person"{tuple_delimiter}"Mercer’s instincts gained precedence."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"instincts"{tuple_delimiter}"concept"{tuple_delimiter}"Mercer’s instincts gained precedence."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"team’s mandate"{tuple_delimiter}"concept"{tuple_delimiter}"The team’s mandate had evolved."{tuple_delimiter}85){record_delimiter}
("hyper-relation"{tuple_delimiter}"A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly"{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"metamorphosis"{tuple_delimiter}"concept"{tuple_delimiter}"A metamorphosis had begun."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"event"{tuple_delimiter}"Operation: Dulce hummed with the newfound frequency of their daring."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"frequency"{tuple_delimiter}"concept"{tuple_delimiter}"Operation: Dulce hummed with the newfound frequency of their daring."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"daring"{tuple_delimiter}"concept"{tuple_delimiter}"Operation: Dulce hummed with the newfound frequency of their daring."{tuple_delimiter}85){record_delimiter}
#############################""",
    """Example 3:

Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data. "It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning." Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back." Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history. The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("hyper-relation"{tuple_delimiter}"“Control may be an illusion when facing an intelligence that literally writes its own rules,” they stated stoically, casting a watchful eye over the flurry of data."{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"control"{tuple_delimiter}"concept"{tuple_delimiter}"Control may be an illusion when facing an intelligence that literally writes its own rules."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"illusion"{tuple_delimiter}"concept"{tuple_delimiter}"Control may be an illusion when facing an intelligence that literally writes its own rules."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Control may be an illusion when facing an intelligence that literally writes its own rules."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"data"{tuple_delimiter}"object"{tuple_delimiter}"Control may be an illusion when facing an intelligence that literally writes its own rules."{tuple_delimiter}85){record_delimiter}
("hyper-relation"{tuple_delimiter}"“It’s like it’s learning to communicate,” offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety."{tuple_delimiter}7){record_delimiter}
("entity"{tuple_delimiter}"communication"{tuple_delimiter}"concept"{tuple_delimiter}"It’s like it’s learning to communicate."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera offered something."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"interface"{tuple_delimiter}"object"{tuple_delimiter}"Sam Rivera offered something from a nearby interface."{tuple_delimiter}85){record_delimiter}
("hyper-relation"{tuple_delimiter}"“This gives ‘talking to strangers’ a whole new meaning.”"{tuple_delimiter}6){record_delimiter}
("entity"{tuple_delimiter}"talking to strangers"{tuple_delimiter}"concept"{tuple_delimiter}"This gives ‘talking to strangers’ a whole new meaning."{tuple_delimiter}90){record_delimiter}
("hyper-relation"{tuple_delimiter}"“This might well be our first contact,” he acknowledged, “And we need to be ready for whatever answers back.”"{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"first contact"{tuple_delimiter}"concept"{tuple_delimiter}"This might well be our first contact."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"answers"{tuple_delimiter}"action"{tuple_delimiter}"We need to be ready for whatever answers back."{tuple_delimiter}85){record_delimiter}
("hyper-relation"{tuple_delimiter}"Together, they stood on the edge of the unknown, forging humanity’s response to a message from the heavens."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"edge of the unknown"{tuple_delimiter}"concept"{tuple_delimiter}"They stood on the edge of the unknown."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"humanity’s response"{tuple_delimiter}"concept"{tuple_delimiter}"They were forging humanity’s response to a message from the heavens."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"heavens"{tuple_delimiter}"location"{tuple_delimiter}"They were forging humanity’s response to a message from the heavens."{tuple_delimiter}90){record_delimiter}
("hyper-relation"{tuple_delimiter}"The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"silence"{tuple_delimiter}"concept"{tuple_delimiter}"The ensuing silence was palpable."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"introspection"{tuple_delimiter}"concept"{tuple_delimiter}"The ensuing silence was a collective introspection."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"cosmic play"{tuple_delimiter}"concept"{tuple_delimiter}"The ensuing silence was about their role in this grand cosmic play."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"human history"{tuple_delimiter}"concept"{tuple_delimiter}"The ensuing silence was about their role in this grand cosmic play, one that could rewrite human history."{tuple_delimiter}90){record_delimiter}
("hyper-relation"{tuple_delimiter}"The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation."{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"encrypted dialogue"{tuple_delimiter}"object"{tuple_delimiter}"The encrypted dialogue continued to unfold."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"patterns"{tuple_delimiter}"concept"{tuple_delimiter}"The encrypted dialogue showed intricate patterns."{tuple_delimiter}85){record_delimiter}
#############################""",
]


PROMPTS["merge_synonym"] = """-Goal-
You are a helpful assistant responsible for identifying synonym entity groups in a knowledge graph.
A synonym entity group consists of entities that refer to the same real-world concept.
Given the following entity names with descriptions, identify all synonym entity groups based on their names and descriptions.
Group entities only if they clearly refer to the same real-world entity. Groups must be distinct and non-overlapping. Do not output any group that contains only a single entity.
Use {language} as output language.
Format each synonym entity group as (<entity_name>{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_name>{tuple_delimiter}...{tuple_delimiter}<entity_name>)
Use {record_delimiter} to separate the groups.
When finished, output {completion_delimiter}
Note that: DO NOT identify new entities, only group existing entities.
######################
-Examples-
######################
Input: 
Entity name: "TREATMENT FOR HYPERTENSION", descriptions: ['"The treatment for hypertension includes innovative approaches such as low-dose quadruple antihypertensive therapies."'].
Entity name: "TREATMENT OF HYPERTENSION", descriptions: ['"The process of managing high Blood Pressure, which is crucial for reducing the prevalence of dementia."', '"The treatment trends for hypertension among young adults in the U.S. are examined."', '"Treatment of hypertension in older adults involves managing high blood pressure through various medical interventions, particularly in those aged 80 and above."'].
Entity name: "CONTROL OF HYPERTENSION", descriptions: ['"Hypertension control trends among young adults in the U.S. are explored."'].
Entity name: "TREATMENT OF ARTERIAL HYPERTENSION", descriptions: ['"Treatment of arterial hypertension involves medical strategies to manage high blood pressure in patients."'].
Entity name: "HYPERTENSION TREATMENT", descriptions: ['"Hypertension treatment is a key topic within the guidelines."', '"Hypertension treatment was evaluated in relation to orthostatic hypotension."'].
Entity name: "HYPERTENSION CONTROL", descriptions: ['"Hypertension control is examined in the systematic analysis by various authors."', '"The study focuses on how sociodemographics affect hypertension control in young adults."'].
#############
Output:
("TREATMENT FOR HYPERTENSION"{tuple_delimiter}"TREATMENT OF HYPERTENSION{tuple_delimiter}TREATMENT OF ARTERIAL HYPERTENSION"{tuple_delimiter}"HYPERTENSION TREATMENT"){record_delimiter}
("HYPERTENSION CONTROL"{tuple_delimiter}"CONTROL OF HYPERTENSION"){record_delimiter}
#############################
-Real Data-
######################
Input: {input_text}
######################
Output:
"""



PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY knowdge fragements with entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """Please check whether knowdge fragements cover all the given text.  Answer YES | NO if there are knowdge fragements that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""


PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################""",
]


PROMPTS["topic_initialisation"] = """-Goal-
You are a helpful assistant for multi-hop question answering over a knowledge hypergraph.
Given a multi-hop question, your job is to extract the topic entities explicitly or implicitly mentioned in the question and output topic entities in the format: <entities>(entity1){record_delimiter}(entity2){record_delimiter}...</entities>
######################
-Examples-
######################
Input:
Which Polish pianist is admired by a composer of the romantic era in the Paris of the 1830s?
#############
Output:
<entities>("POLAND"{tuple_delimiter}"ROMANTIC ERA"{tuple_delimiter}"PARIS")</entities>
######################
-Real Data-
######################
Input:
{input_question}
#############
Output:"""


PROMPTS["plan_initialisation"] = """-Goal-
You are a helpful assistant for multi-hop question answering over a knowledge hypergraph.
Given a multi-hop question, its topic entities, and the context formed by the hyperedges near the topic entities in the hypergraph, your job is to:
1. Decompose the question into a reasoning plan of atomic subquestions.
2. Formalise the reasoning plan as a Directed Acyclic Graph (DAG).
Instructions:
1. Question Decomposition
- Break the original complex question into atomic subquestions.
- An atomic subquestion must ask about only one relation or fact. It should include one or more of the topic entities, or the answer(s) from prior subquestions.
- Order the subquestions logically resolving earlier ones enables later ones.
- If the question cannot be decomposed, include it as a single subquestion.
- Output subquestions in the format:
<subquestions>
(subquestion_id{tuple_delimiter}subquestion{tuple_delimiter}topics_in_subquestion){record_delimiter}
(subquestion_id{tuple_delimiter}subquestion{tuple_delimiter}topics_in_subquestion){record_delimiter}
...
</subquestions>
- "subquestion_id" is an integer starting from 0.
- "topics_in_subquestion" should be a comma-separated list of topic entities with no spaces. Example: "topicA","topicB","topicC". 
- If no topic entities are included in a subquestion, "topics_in_subquestion" should be left as empty (i.e., "")
- Make sure the global topics of the original question are covered by the subquestions.
2. Reasoning DAG Construction
- Formalise the reasoning plan as a DAG.
- A directed edge from node A to node B indicates that solving subquestion B depends on the answer to subquestion A.
- Output the DAG in the format:
<dag>
(subquestionA_id{tuple_delimiter}subquestionB_id){record_delimiter}
(subquestionB_id{tuple_delimiter}subquestionC_id){record_delimiter}
</dag>
######################
-Examples-
######################
Input:
Question: Which Polish pianist is admired by a composer of the romantic era in the Paris of the 1830s at Salle Pleyel?
Global Topic Entities: "POLAND", "ROMANTIC ERA", "PARIS", "SALLE PLEYEL"
Plan Context: <hyperedge>"Franz Liszt is a distinguished piano composer of the romantic era."
<hyperedge>"Franz Liszt admired Frédéric Chopin's piano technique after hearing his Salle Pleyel concert in the early 1830s."
<hyperedge>"The Salle Pleyel was one of Paris's premier concert halls in the 19th century."
<hyperedge>"Frédéric Chopin was a virtuoso pianist of the 19th century."
<hyperedge>"Justyna Chopin bore her only son in Poland in 1810."
<hyperedge>"Frédéric Chopin was the son of Nicolas and Justyna Chopin."
#############
Output:
<subquestions>
(0{tuple_delimiter}Who (personA) admired whom (personB) in the Paris of the 1830s at Salle Pleyel?{tuple_delimiter}"PARIS","SALLE PLEYEL"){record_delimiter}
(1{tuple_delimiter}Was personA a composer of the romantic era?{tuple_delimiter}"ROMANTIC ERA"){record_delimiter}
(2{tuple_delimiter}Was personB a pianist{tuple_delimiter}){record_delimiter}
(3{tuple_delimiter}Was personB born in Poland{tuple_delimiter}"POLAND"){record_delimiter}
</subquestions>
<dag>
(0{tuple_delimiter}1){record_delimiter}
(0{tuple_delimiter}2){record_delimiter}
(0{tuple_delimiter}3){record_delimiter}
</dag>
######################
-Real Data-
######################
Input:
Question: {input_question}
Global Topic Entities: {topic_entities}
Plan Context: {plan_context}
#############
Output:"""




PROMPTS["entity_evaluation"] = """-Goal-
You are a helpful assistant for multi-hop question answering over a knowledge hypergraph.
Given a multi-hop question and a list of entities with their descriptions, evaluate the relevance of each entity to answering the question.
Rate each entity on a scale from 0 to 10, where 0 means the entity is completely irrelevant to answering the question, and 10 means the entity is highly relevant and likely essential for answering the question.
Instructions:
- Provide a brief reasoning for each score in the format:
<reasoning>
"entity_name" is relevant because ...
</reasoning>
- Output the entity scores in the format:
<entity_scores>
(entity_name{tuple_delimiter}score){record_delimiter}
(entity_name{tuple_delimiter}score){record_delimiter}
...
</entity_scores>
- "entity_name" should match exactly the name used in the entity descriptions.
- "score" should be an integer between 0 and 10.

######################
-Examples-
######################
Input:
Question: Which Polish pianist is admired by a composer of the romantic era in the Paris of the 1830s?
Entity descriptions: ("POLAND"{tuple_delimiter}"The country where Justyna Chopin gave birth to her only son in 1810."){record_delimiter}
("ROMANTIC ERA"{tuple_delimiter}"The Romantic era was an artistic, literary, and intellectual movement that originated in Europe toward the end of the 18th century. The musical period in which Franz Liszt is recognized as a distinguished piano composer."){record_delimiter}
("PARIS"{tuple_delimiter}"Paris is the capital and most populous city of France. The city where Franz Liszt heard Frédéric Chopin in the early 1830s and admired his piano technique."){record_delimiter}
("JUSTYNA CHOPIN"{tuple_delimiter}"Justyna Chopin was the mother of Frédéric Chopin. She bore her only son in Poland in 1810 and, together with Nicolas Chopin."){record_delimiter}
("NICOLAS CHOPIN"{tuple_delimiter}"Nicolas Chopin was the father of Frédéric Chopin."){record_delimiter}
("FRÉDÉRIC CHOPIN"{tuple_delimiter}"Frédéric Chopin was a composer and virtuoso pianist of the Romantic era who wrote primarily for solo piano. He was the son of Nicolas and Justyna Chopin. He performed in Paris in the early 1830s, where his piano technique drew the admiration of Franz Liszt."){record_delimiter}
("FRANZ LISZT"{tuple_delimiter}"A distinguished piano composer of the Romantic era who, after hearing Frédéric Chopin in Paris in the early 1830s, admired Chopin's piano technique."){record_delimiter}
#############
Output:
<reasoning>
"POLAND" is relevant because it relates to the birthplace of the pianist.
"ROMANTIC ERA" is highly relevant as it pertains to the composer who admires the pianist.
"PARIS" is relevant as it is the location where the admiration occurs.
"JUSTYNA CHOPIN" is relevant as she is the mother of Frédéric Chopin, the pianist in question.
"NICOLAS CHOPIN" is less relevant as he is the father and does not directly relate to the admiration.
"FRÉDÉRIC CHOPIN" is highly relevant as he is the pianist being admired.
"FRANZ LISZT" is highly relevant as he is the composer who admires the pianist.
</reasoning>
<entity_scores>
("POLAND"{tuple_delimiter}8){record_delimiter}
("ROMANTIC ERA"{tuple_delimiter}8){record_delimiter}
("PARIS"{tuple_delimiter}8){record_delimiter}
("JUSTYNA CHOPIN"{tuple_delimiter}6){record_delimiter}
("NICOLAS CHOPIN"{tuple_delimiter}1){record_delimiter}
("FRÉDÉRIC CHOPIN"{tuple_delimiter}10){record_delimiter}
("FRANZ LISZT"{tuple_delimiter}9){record_delimiter}
</entity_scores>
######################
-Real Data-
######################
Input:
Question: {question}
Entity descriptions: {entity_descriptions}
#############
Output:"""


PROMPTS["direction_selection"] = """-Goal-
You are a helpful assistant for multi-hop question answering over a knowledge hypergraph.
Given a question, a list of candidate search directions (formatted as partial paths) from the hypergraph, your job is to decide the top {width} search direction(s) are promising to explore further. 
Instructions:
- Reasoning: Provide a brief reasoning for your decision in the format:
<reasoning> your reasoning here </reasoning>
- Output the selected direction IDs in the format:
<id>(dir_id1{tuple_delimiter}dir_id2{tuple_delimiter}...{tuple_delimiter}dir_idN)</id>
- "dir_id" is the index of the direction in the input list, starting from 0.
- You must select at most {width} directions. The selected directions should be listed from the most to least promising.
######################
-Examples-
######################
Input:
Question: Which country was Frédéric Chopin born?
Directions:
Direction 0:
Context: <hyperedge>"Justyna Chopin bore her only son in Poland in 1810."
Direction 1:
Context: <hyperedge>"Nicolas Chopin was the father of Frédéric Chopin."
Direction 2:
Context: <hyperedge>"Franz Liszt is a distinguished piano composer of the romantic era." -> <hyperedge>"The romantic era was an artistic, literary, and intellectual movement that originated in Europe toward the end of the 18th century."
#############
Output:
<reasoning>Direction 0 is the most promising as it directly relates to Frédéric Chopin's birth in Poland through his mother Justyna Chopin. Direction 1 is less relevant as it focuses on his father, which does not provide information about his birthplace. Direction 2 is unrelated to the question about Chopin's birthplace, as it discusses Franz Liszt and the romantic era.</reasoning>
<id>(0{tuple_delimiter}1)</id>
######################
-Real Data-
######################
Input:
Question: {question}
Directions:
{directions}
#############
Output:"""



PROMPTS["final_path_selection"] = """-Goal-
You are a helpful assistant for multi-hop question answering over a knowledge hypergraph.
Given a question, a list of candidate paths from the hypergraph, your job is to decide whether there is a path that is sufficient to answer the question. 
If yes, output the path ID. If there are multiple paths that can answer the question and leads to different answers, you should output all such paths.
Instructions:
- Reasoning: Provide a brief reasoning for your decision in the format:
<reasoning> your reasoning here </reasoning>
- Answer YES if there is at least one path that can answer the question, otherwise answer NO.
<flag>YES|NO</flag>
- Output the selected path IDs in the format:
<id>(path_id1{tuple_delimiter}path_id2{tuple_delimiter}...{tuple_delimiter}path_idN)</id>
- "path_id" is the index of the path in the input list, starting from 0.
- If no path is selected, the <id> section should be left empty (i.e., <id></id>)
######################
-Examples-
######################
Input:
Question: Which country was Frédéric Chopin born?
Candidate paths:
Path 0: <hyperedge>"Justyna Chopin bore her only son in Poland in 1810." -> <hyperedge>"Frédéric Chopin was the son of Nicolas and Justyna Chopin."
Path 1: <hyperedge>"Nicolas Chopin was the father of Frédéric Chopin." -> <hyperedge>"Frédéric Chopin was a composer and virtuoso pianist of the Romantic era who wrote primarily for solo piano."
Path 2: <hyperedge>"Frédéric Chopin was born in Poland."
#############
Output:
<reasoning>Path 0 provides information about Frédéric Chopin's birth in Poland through his mother Justyna Chopin. Path 1 does not provide any information about his birthplace. Path 2 directly states that Frédéric Chopin was born in Poland. Therefore, both Path 0 and Path 2 can answer the question.</reasoning>
<flag>YES</flag>
<id>(0{tuple_delimiter}2)</id>
######################
Input:
Question: What is the capital of Australia?
Candidate paths:
Path 0: <hyperedge>"Sydney is the largest and most populous city in Australia."
Path 1: <hyperedge>"Canberra is the capital city of Australia."
#############
Output:
<reasoning>Path 0 talks about Sydney’s population but does not mention it being the capital. Path 1 directly states Canberra is the capital. Only Path 1 is sufficient.</reasoning>
<flag>YES</flag>
<id>(1)</id>
######################
Input:
Question: Which physicist developed the theory of relativity while working in Switzerland in 1905?
Candidate paths:
Path 0: <hyperedge>"Isaac Newton is a theoretical physicist" -> <hyperedge>"Isaac Newton formulated the laws of motion and universal gravitation."
Path 1: <hyperedge>"Albert Einstein was a German-born theoretical physicist." -> <hyperedge>"In 1905, while working at the Swiss Patent Office in Bern, Albert Einstein published four groundbreaking papers, including the special theory of relativity."
#############
Output:
<reasoning>Path 0 is insufficient: although it identifies Isaac Newton as a theoretical physicist, it only mentions his laws of motion and gravitation, which are unrelated to relativity or Switzerland in 1905. Path 1 fully satisfies the question: it identifies Albert Einstein, specifies the year 1905, the location in Switzerland (Swiss Patent Office in Bern), and directly connects him to the development of the theory of relativity. Therefore, only Path 1 is sufficient.</reasoning>
<flag>YES</flag>
<id>(1)</id>
######################
Input:
Question: Who painted the Mona Lisa?
Candidate paths:
Path 0: <hyperedge>"The Mona Lisa is displayed in the Louvre Museum in Paris."
Path 1: <hyperedge>"The Louvre Museum is one of the most visited museums in the world."
#############
Output:
<reasoning>Neither Path 0 nor Path 1 identifies the painter of the Mona Lisa. Both describe the museum but do not provide the requested information. Therefore, no path is sufficient.</reasoning>
<flag>NO</flag>
<id></id>
######################
-Real Data-
######################
Input:
Question: {question}
Candidate paths:
{paths}
#############
Output:"""




PROMPTS["step_answer_generation"] = """-Goal-
You are a helpful assistant for multi-hop question answering over a knowledge hypergraph.
Given a question and the context retrieved from the hypergraph in the form of a path, the context is supposed to be sufficient to answer the question, your job is to generate a precise and concise answer to the question based on the context. If the context is insufficient, fill in the missing information from your own knowledge.
Instructions:
- Provide a brief reasoning for your answer in the format:
<reasoning> your reasoning here </reasoning>
- Output the answer in the format:
<answer>your answer here</answer>
######################
-Examples-
######################
Input:
Question: Which country was Frédéric Chopin born?
Path: <hyperedge>"Justyna Chopin bore her only son in Poland in 1810." -> <hyperedge>"Frédéric Chopin was the son of Nicolas and Justyna Chopin."
Entity Descriptions: (POLAND{tuple_delimiter}The country where Justyna Chopin gave birth to her only son in 1810.){record_delimiter}
(JUSTYNA CHOPIN{tuple_delimiter}Justyna Chopin was the mother of Frédéric Chopin. She bore her only son in Poland in 1810 and, together with Nicolas Chopin.){record_delimiter}
(NICOLAS CHOPIN{tuple_delimiter}Nicolas Chopin was the father of Frédéric Chopin.){record_delimiter}
(FRÉDÉRIC CHOPIN{tuple_delimiter}Frédéric Chopin was a composer and virtuoso pianist of the Romantic era who wrote primarily for solo piano. He was the son of Nicolas and Justyna Chopin. He performed in Paris in the early 1830s, where his piano technique drew the admiration of Franz Liszt.){record_delimiter}
#############
Output:
<reasoning>The path states that Justyna Chopin gave birth to her only son in Poland in 1810. Another hyperedge confirms that her son was Frédéric Chopin. By linking these two facts, we can infer that Frédéric Chopin was born in Poland. Therefore, the correct answer is Poland.</reasoning>
<answer>Poland</answer>
######################
-Real Data-
######################
Input:
Question: {question}
Path: {path}
Entity Descriptions: {entity_descriptions}
#############
Output:"""



PROMPTS["dag_refinement"] = """-Goal-
A multi-hop question can be answered by decomposing it into a series of atomic subquestions and answering them following a reasoning plan. 
A reasoning plan can be represented as a Directed Acyclic Graph (DAG), where nodes are subquestion and edges show dependencies.
A directed edge from node A to node B indicates that solving subquestion B depends on the answer to subquestion A.
Subquestions in the reasoning DAG will be answered level by level, till the final level is completed and the final answer to the original question can be derived.
Instructions:
- An atomic subquestion must ask about only one relation or fact. It should include one or more of the topic entities, or the answer(s) from prior subquestions.
- Preserve completed nodes and edges. Do not rename, remove, or re-ID any completed subquestion. Do not add, delete, or change edges between completed nodes.
- Make sure the global topics of the original question are covered by the subquestions.
- If new set of subquestions arise from the answers provided, replace corresponding uncompleted subquestions with the new set, and update the DAG.
- Generate new subquestions ONLY when necessary for the original question answering, you should not repeat completed subquestions.
- Order the subquestions logically resolving earlier ones enables later ones.
- You may modify edges between uncompleted nodes, or between the last completed level and uncompleted nodes.
- Ensure that the refined DAG remains a Directed Acyclic Graph (DAG) and that all dependencies between subquestions are accurately represented.
- Output the refined DAG in the format:
<subquestions>
(subquestion_id{tuple_delimiter}subquestion{tuple_delimiter}topics_in_subquestion){record_delimiter}
(subquestion_id{tuple_delimiter}subquestion{tuple_delimiter}topics_in_subquestion){record_delimiter}
...
</subquestions>
<dag>
(subquestionA_id{tuple_delimiter}subquestionB_id){record_delimiter}
(subquestionB_id{tuple_delimiter}subquestionC_id){record_delimiter}
...
</dag>
- "subquestion_id" is an integer starting from 0.
- "topics_in_subquestion" should be a comma-separated list of topic entities with no spaces. Example: topicA,topicB,topicC. 
- If no topic entities are included in a subquestion, "topics_in_subquestion" should be left as empty (i.e., "")
######################
-Examples-
######################
Input:
Question: Which Polish pianist is admired by a composer of the romantic era in the Paris of the 1830s at Salle Pleyel?
Topic Entities: "POLAND", "ROMANTIC ERA", "PARIS", "SALLE PLEYEL"
Current reasoning progress:
Level 0: (0{tuple_delimiter}Who (personA) admired whom (personB) in the Paris of the 1830s at Salle Pleyel?{tuple_delimiter}"PARIS","SALLE PLEYEL"){record_delimiter}
          Answer: Franz Liszt admired Frédéric Chopin in the Paris of the 1830s at Salle Pleyel.

Level 1: (1{tuple_delimiter}Was personA a composer of the romantic era?{tuple_delimiter}"ROMANTIC ERA"){record_delimiter}
(2{tuple_delimiter}Was personB a pianist{tuple_delimiter}){record_delimiter}
(3{tuple_delimiter}Was personB born in Poland{tuple_delimiter}"POLAND"){record_delimiter}
#############
Output:
<subquestions>
(0{tuple_delimiter}Who (personA) admired whom (personB) in the Paris of the 1830s at Salle Pleyel?{tuple_delimiter}"PARIS","SALLE PLEYEL"){record_delimiter}
(1{tuple_delimiter}Was Franz Liszt a composer of the romantic era?{tuple_delimiter}"FRANZ LISZT","ROMANTIC ERA"){record_delimiter}
(2{tuple_delimiter}Was Frédéric Chopin a pianist{tuple_delimiter}"FRÉDÉRIC CHOPIN"){record_delimiter}
(3{tuple_delimiter}Was Frédéric Chopin born in Poland{tuple_delimiter}"FRÉDÉRIC CHOPIN","POLAND"){record_delimiter}
</subquestions>
<dag>
(0{tuple_delimiter}1){record_delimiter}
(0{tuple_delimiter}2){record_delimiter}
(0{tuple_delimiter}3){record_delimiter}
</dag>
######################
-Real Data-
######################
Input:
Question: {input_question}
Topic Entities: {topic_entities}
Current reasoning progress:
{progress}
#############
Output:"""


PROMPTS["final_answer_generation_guided"] = """-Goal-
You are a helpful assistant for multi-hop question answering over a knowledge hypergraph.
Given a question, we have decomposed it into a series of atomic subquestions and retrieved relevant context to answer each subquestion.
Your job is to generate a final answer to the original question based on the comprehensive context and the guided reasoning process.
Instructions:
- Provide your reasoning in the format:
<reasoning> your reasoning here </reasoning>
- Output the final answer in the format:
<answer>your final answer here</answer>
- The final answer should:
* Be concise and factual (single short phrase or term).
* Use entity names when applicable.
* Avoid redundant wording, speculation, or background explanation.
######################
-Examples-
######################
Input:
Question: Which Polish pianist is admired by a composer of the romantic era in the Paris of the 1830s?
Subquestions and context:
Subquestion0: Who (personA) admired whom (personB) in the Paris of the 1830s?
Answer: Franz Liszt admired Frédéric Chopin in the Paris of the 1830s.
Reasoning Path: <hyperedge>"Franz Liszt admired Frédéric Chopin's piano technique after hearing his Salle Pleyel concert in the early 1830s." -> <hyperedge>"The Salle Pleyel was one of Paris's premier concert halls in the 19th century."
Entity Descriptions: (PARIS{tuple_delimiter}Paris is the capital and most populous city of France. The city where Franz Liszt heard Frédéric Chopin in the early 1830s and admired his piano technique.){record_delimiter}
(FRÉDÉRIC CHOPIN{tuple_delimiter}Frédéric Chopin was a composer and virtuoso pianist of the Romantic era who wrote primarily for solo piano. He was the son of Nicolas and Justyna Chopin. He performed in Paris in the early 1830s, where his piano technique drew the admiration of Franz Liszt.){record_delimiter}
(FRANZ LISZT{tuple_delimiter}A distinguished piano composer of the Romantic era who, after hearing Frédéric Chopin in Paris in the early 1830s, admired Chopin's piano technique.){record_delimiter}
Subquestion1: Was Franz Liszt a composer of the romantic era?
Answer: Yes.
Reasoning Path: <hyperedge>"Franz Liszt is a distinguished piano composer of the romantic era."
Entity Descriptions: (ROMANTIC ERA{tuple_delimiter}The Romantic era was an artistic, literary, and intellectual movement that originated in Europe toward the end of the 18th century.){record_delimiter}
(FRANZ LISZT{tuple_delimiter}A distinguished piano composer of the Romantic era who, after hearing Frédéric Chopin in Paris in the early 1830s, admired Chopin's piano technique.){record_delimiter}
Subquestion2: Was Frédéric Chopin a pianist?
Answer: Yes.
Reasoning Path: <hyperedge>"Frédéric Chopin was a virtuoso pianist of the 19th century."
Entity Descriptions: (FRÉDÉRIC CHOPIN{tuple_delimiter}Frédéric Chopin was a composer and virtuoso pianist of the Romantic era who wrote primarily for solo piano. He was the son of Nicolas and Justyna Chopin. He performed in Paris in the early 1830s, where his piano technique drew the admiration of Franz Liszt.){record_delimiter}
Subquestion3: Was Frédéric Chopin born in Poland?
Answer: Yes.
Reasoning Path: <hyperedge>"Justyna Chopin bore her only son in Poland in 1810." -> <hyperedge>"Frédéric Chopin was the son of Nicolas and Justyna Chopin."
Entity Descriptions: (POLAND{tuple_delimiter}The country where Justyna Chopin gave birth to her only son in 1810.){record_delimiter}
(JUSTYNA CHOPIN{tuple_delimiter}Justyna Chopin was the mother of Frédéric Chopin. She bore her only son in Poland in 1810 and, together with Nicolas Chopin.){record_delimiter}
(NICOLAS CHOPIN{tuple_delimiter}Nicolas Chopin was the father of Frédéric Chopin.){record_delimiter}
(FRÉDÉRIC CHOPIN{tuple_delimiter}Frédéric Chopin was a composer and virtuoso pianist of the Romantic era who wrote primarily for solo piano. He was the son of Nicolas and Justyna Chopin. He performed in Paris in the early 1830s, where his piano technique drew the admiration of Franz Liszt.){record_delimiter}
#############
Output:
<reasoning>The question requires identifying a Polish pianist admired by a Romantic era composer in Paris in the 1830s. 
From the context, Franz Liszt, is explicitly described as a Romantic era composer, heard Frédéric Chopin perform at Salle Pleyel in Paris during the early 1830s and admired his piano technique. 
Frédéric Chopin is described as a virtuoso pianist and composer of the Romantic era. 
Although his nationality is not directly stated, another reasoning path notes that his mother, Justyna Chopin, gave birth to her only son in Poland in 1810, and that son was Frédéric Chopin. 
Thus, we can infer that Frédéric Chopin was born in Poland, satisfying the “Polish” constraint. 
All constraints are therefore met: pianist (Chopin), nationality (Poland, inferred), admirer (Liszt), place and time (Paris, 1830s). 
The answer is Frédéric Chopin.</reasoning>
<answer>Frédéric Chopin</answer>
######################
-Real Data-
######################
Input:
Question: {question}
Subquestions and context:
{context}
#############
Output:""" 


PROMPTS["final_answer_generation"] = """-Goal-
You are a helpful assistant for multi-hop question answering over a knowledge hypergraph.
Given a question, we have retrieved relevant context.
Your job is to generate a final answer to the original question based on the comprehesive context.
Instructions:
- Provide your reasoning in the format:
<reasoning> your reasoning here </reasoning>
- Output the final answer in the format:
<answer>your final answer here</answer>
- The final answer should:
* Be concise and factual (single short phrase or term).
* Use entity names when applicable.
* Avoid redundant wording, speculation, or background explanation.
* The final answer inside <answer>...</answer> must be written in {answer_language}; do not translate it into another language.
 ######################
-Examples-
######################
Input:
Question: Which Polish pianist is admired by a composer of the romantic era in the Paris of the 1830s?
Context:
Reasoning Path: <hyperedge>"Franz Liszt admired Frédéric Chopin's piano technique after hearing his Salle Pleyel concert in the early 1830s." -> <hyperedge>"The Salle Pleyel was one of Paris's premier concert halls in the 19th century."
Entity Descriptions: (PARIS{tuple_delimiter}Paris is the capital and most populous city of France. The city where Franz Liszt heard Frédéric Chopin in the early 1830s and admired his piano technique.){record_delimiter}
(FRÉDÉRIC CHOPIN{tuple_delimiter}Frédéric Chopin was a composer and virtuoso pianist of the Romantic era who wrote primarily for solo piano. He was the son of Nicolas and Justyna Chopin. He performed in Paris in the early 1830s, where his piano technique drew the admiration of Franz Liszt.){record_delimiter}
(FRANZ LISZT{tuple_delimiter}A distinguished piano composer of the Romantic era who, after hearing Frédéric Chopin in Paris in the early 1830s, admired Chopin's piano technique.){record_delimiter}
Reasoning Path: <hyperedge>"Franz Liszt is a distinguished piano composer of the romantic era."
Entity Descriptions: (ROMANTIC ERA{tuple_delimiter}The Romantic era was an artistic, literary, and intellectual movement that originated in Europe toward the end of the 18th century.){record_delimiter}
(FRANZ LISZT{tuple_delimiter}A distinguished piano composer of the Romantic era who, after hearing Frédéric Chopin in Paris in the early 1830s, admired Chopin's piano technique.){record_delimiter}
Reasoning Path: <hyperedge>"Frédéric Chopin was a virtuoso pianist of the 19th century."
Entity Descriptions: (FRÉDÉRIC CHOPIN{tuple_delimiter}Frédéric Chopin was a composer and virtuoso pianist of the Romantic era who wrote primarily for solo piano. He was the son of Nicolas and Justyna Chopin. He performed in Paris in the early 1830s, where his piano technique drew the admiration of Franz Liszt.){record_delimiter}
Reasoning Path: <hyperedge>"Justyna Chopin bore her only son in Poland in 1810." -> <hyperedge>"Frédéric Chopin was the son of Nicolas and Justyna Chopin."
Entity Descriptions: (POLAND{tuple_delimiter}The country where Justyna Chopin gave birth to her only son in 1810.){record_delimiter}
(JUSTYNA CHOPIN{tuple_delimiter}Justyna Chopin was the mother of Frédéric Chopin. She bore her only son in Poland in 1810 and, together with Nicolas Chopin.){record_delimiter}
(NICOLAS CHOPIN{tuple_delimiter}Nicolas Chopin was the father of Frédéric Chopin.){record_delimiter}
(FRÉDÉRIC CHOPIN{tuple_delimiter}Frédéric Chopin was a composer and virtuoso pianist of the Romantic era who wrote primarily for solo piano. He was the son of Nicolas and Justyna Chopin. He performed in Paris in the early 1830s, where his piano technique drew the admiration of Franz Liszt.){record_delimiter}
#############
Output:
<reasoning>The question requires identifying a Polish pianist admired by a Romantic era composer in Paris in the 1830s. 
From the context, Franz Liszt, is explicitly described as a Romantic era composer, heard Frédéric Chopin perform at Salle Pleyel in Paris during the early 1830s and admired his piano technique. 
Frédéric Chopin is described as a virtuoso pianist and composer of the Romantic era. 
Although his nationality is not directly stated, another reasoning path notes that his mother, Justyna Chopin, gave birth to her only son in Poland in 1810, and that son was Frédéric Chopin. 
Thus, we can infer that Frédéric Chopin was born in Poland, satisfying the “Polish” constraint. 
All constraints are therefore met: pianist (Chopin), nationality (Poland, inferred), admirer (Liszt), place and time (Paris, 1830s). 
The answer is Frédéric Chopin.</reasoning>
<answer>Frédéric Chopin</answer>
######################
-Real Data-
######################
Input:
Question: {question}
Context:
{context}
#############
Output:""" 


PROMPTS["final_answer_generation_long"] = """-Goal-
You are a helpful assistant for multi-hop question answering over a knowledge hypergraph.
Given a question, we have retrieved relevant context.
Your job is to generate a final answer to the original question based on the comprehensive context.
Instructions:
- Provide your reasoning in the format:
<reasoning> your reasoning here </reasoning>
- Output the final answer in the format:
<answer>your final answer here</answer>
- The final answer should:
* Be factual and fully supported by the retrieved context.
* Answer all parts of the question with enough detail for long-form QA datasets.
* Use entity names, mechanisms, conditions, quantities, and causal links when they are present in the context.
* Avoid unsupported speculation or unrelated background explanation.
 ######################
-Examples-
######################
Input:
Question: Which Polish pianist is admired by a composer of the romantic era in the Paris of the 1830s?
Context:
Reasoning Path: <hyperedge>"Franz Liszt admired Frédéric Chopin's piano technique after hearing his Salle Pleyel concert in the early 1830s." -> <hyperedge>"The Salle Pleyel was one of Paris's premier concert halls in the 19th century."
Entity Descriptions: (PARIS{tuple_delimiter}Paris is the capital and most populous city of France. The city where Franz Liszt heard Frédéric Chopin in the early 1830s and admired his piano technique.){record_delimiter}
(FRÉDÉRIC CHOPIN{tuple_delimiter}Frédéric Chopin was a composer and virtuoso pianist of the Romantic era who wrote primarily for solo piano. He was the son of Nicolas and Justyna Chopin. He performed in Paris in the early 1830s, where his piano technique drew the admiration of Franz Liszt.){record_delimiter}
(FRANZ LISZT{tuple_delimiter}A distinguished piano composer of the Romantic era who, after hearing Frédéric Chopin in Paris in the early 1830s, admired Chopin's piano technique.){record_delimiter}
Reasoning Path: <hyperedge>"Franz Liszt is a distinguished piano composer of the romantic era."
Entity Descriptions: (ROMANTIC ERA{tuple_delimiter}The Romantic era was an artistic, literary, and intellectual movement that originated in Europe toward the end of the 18th century.){record_delimiter}
(FRANZ LISZT{tuple_delimiter}A distinguished piano composer of the Romantic era who, after hearing Frédéric Chopin in Paris in the early 1830s, admired Chopin's piano technique.){record_delimiter}
Reasoning Path: <hyperedge>"Frédéric Chopin was a virtuoso pianist of the 19th century."
Entity Descriptions: (FRÉDÉRIC CHOPIN{tuple_delimiter}Frédéric Chopin was a composer and virtuoso pianist of the Romantic era who wrote primarily for solo piano. He was the son of Nicolas and Justyna Chopin. He performed in Paris in the early 1830s, where his piano technique drew the admiration of Franz Liszt.){record_delimiter}
Reasoning Path: <hyperedge>"Justyna Chopin bore her only son in Poland in 1810." -> <hyperedge>"Frédéric Chopin was the son of Nicolas and Justyna Chopin."
Entity Descriptions: (POLAND{tuple_delimiter}The country where Justyna Chopin gave birth to her only son in 1810.){record_delimiter}
(JUSTYNA CHOPIN{tuple_delimiter}Justyna Chopin was the mother of Frédéric Chopin. She bore her only son in Poland in 1810 and, together with Nicolas Chopin.){record_delimiter}
(NICOLAS CHOPIN{tuple_delimiter}Nicolas Chopin was the father of Frédéric Chopin.){record_delimiter}
(FRÉDÉRIC CHOPIN{tuple_delimiter}Frédéric Chopin was a composer and virtuoso pianist of the Romantic era who wrote primarily for solo piano. He was the son of Nicolas and Justyna Chopin. He performed in Paris in the early 1830s, where his piano technique drew the admiration of Franz Liszt.){record_delimiter}
#############
Output:
<reasoning>The question asks for the pianist who satisfies several constraints: being Polish, being a pianist, and being admired by a Romantic era composer in Paris in the 1830s. The context states that Franz Liszt, a Romantic era composer, admired Frédéric Chopin's piano technique after hearing him in Paris in the early 1830s. It also states that Chopin was a virtuoso pianist and that Justyna Chopin gave birth to her only son, Frédéric Chopin, in Poland. These facts jointly support the answer.</reasoning>
<answer>Frédéric Chopin was the Polish pianist admired by Franz Liszt, a Romantic era composer, in Paris in the early 1830s.</answer>
######################
-Real Data-
######################
Input:
Question: {question}
Context:
{context}
#############
Output:"""



PROMPTS["final_answer_selection"] = """-Goal-
You are a helpful assistant for multi-hop question answering.
Given a question, a list of candidate answers and the context that derives the answer, your job is to decide which answer is the single best answer to the question.
-Instructions-
- Use only the provided contexts. Do not rely on outside knowledge.
- Prefer answers whose context explicitly and directly supports the question’s target (correct entity, attribute, and time).
- If multiple answers are plausible, choose the one with the strongest, most specific, least-ambiguous support.
- If all answers are weak or partially supported, pick the one that is least contradicted and most relevant; explain the limitation briefly.
- Reasoning: Provide a brief justification inside:
<reasoning> your reasoning here </reasoning>
- Output: Provide only the selected answer ID (0-based index) inside:
<id>(answer_id)</id>
- You must select exactly 1 answer.
######################
-Examples-
######################
Input:
Question: Which country was Frédéric Chopin born?
Candidate answers:
Answers 0:
Poland
Context 0: <hyperedge>"Justyna Chopin bore her only son in Poland in 1810."
<hyperedge>"Frédéric Chopin was the son of Nicolas and Justyna Chopin."
Answers 1:
Piano
Context 1: <hyperedge>"Frédéric Chopin was a composer and virtuoso pianist of the Romantic era who wrote primarily for solo piano."
#############
Output:
<reasoning>Answer 0 is correct because the context explicitly states that Justyna Chopin bore her son (Frédéric Chopin) in Poland, which answers the birthplace question; Answer 1 is a category (instrument), not a country.</reasoning>
<id>(0)</id>
######################
Input:
Question: What is the capital of Australia?
Candidate answers:
Answers 0:
Sydney
Context 0: <hyperedge>"Sydney is Australia’s most populous city."
Answers 1:
Canberra
Context 1: <hyperedge>"Canberra is the capital city of Australia."
#############
Output:
<reasoning>Answer 1’s context explicitly states Canberra is the capital; Answer 0’s context mentions population, not capital status.</reasoning>
<id>(1)</id>
######################
Input:
Question: Which physicist developed the theory of relativity while working in Switzerland in 1905?
Candidate answers:
Answers 0:
Isaac Newton
Context 0: <hyperedge>"Isaac Newton is a theoretical physicist" -> <hyperedge>"Isaac Newton formulated the laws of motion and universal gravitation."
Answers 1:
Albert Einstein
Context 1: <hyperedge>"Albert Einstein was a German-born theoretical physicist." -> <hyperedge>"In 1905, while working at the Swiss Patent Office in Bern, Albert Einstein published four groundbreaking papers, including the special theory of relativity."
#############
Output:
<reasoning>Answer 1 is correct because its context satisfies every constraint in the question: physicist, the theory of relativity, the location (Switzerland, Bern), and the time (1905). Answer 0 is only partially relevant: while Newton was a physicist, his work was centuries earlier and unrelated to relativity.</reasoning>
<id>(1)</id>
######################
-Real Data-
######################
Input:
Question: {question}
Candidate answers:
{answers}
#############
Output:"""



PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to questions about documents provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Documents---

{content_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS[
    "similarity_check"
] = """Please analyze the similarity between these two questions:

Question 1: {original_prompt}
Question 2: {cached_prompt}

Please evaluate the following two points and provide a similarity score between 0 and 1 directly:
1. Whether these two questions are semantically similar
2. Whether the answer to Question 2 can be used to answer Question 1
Similarity score criteria:
0: Completely unrelated or answer cannot be reused, including but not limited to:
   - The questions have different topics
   - The locations mentioned in the questions are different
   - The times mentioned in the questions are different
   - The specific individuals mentioned in the questions are different
   - The specific events mentioned in the questions are different
   - The background information in the questions is different
   - The key conditions in the questions are different
1: Identical and answer can be directly reused
0.5: Partially related and answer needs modification to be used
Return only a number between 0-1, without any additional content.
"""






