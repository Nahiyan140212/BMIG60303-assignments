# Assignment 05: Word2Vec Analysis Report
## MIMIC-IV notes (Ophthalmology Focus)

```bash
#note_ids for reproducibility
#note_ids are in the note_ids.txt file
# df = pd.read_csv('discharge.csv')
# def create_subset(df, note_id_list):
#     df_sample = df[df['note_id'].isin(note_ids)]
#     return subset
```

### Dataset Description
This analysis uses medical abstracts from a MIMIC ophthalmology-related ICD code dataset. While the primary complaints in the dataset are not exclusively ophthalmic, the abstracts contain rich medical terminology including anatomical terms, medications, procedures, and conditions related to eye care and general medicine.

```
[('abseos-0.07', 0.863), ('number', 0.858), ('05:43am', 0.857), 
 ('10:10pm', 0.855), ('12:35pm', 0.851)]
```
The poor model completely fails to capture medical relationships, returning timestamps and random strings instead of anatomically related terms. 

**Better Model 1 Results:**
```
[('bathtub', 0.414), ('up', 0.392), ('fistulagram', 0.353), 
 ('anticipation', 0.343), ('extent', 0.342)]
```
While slightly improved, this model still shows significant noise. Only 'fistulagram' (a medical procedure) shows any medical relevance.

**Better Model 2 Results:**
```
[('resorption', 0.487), ('neuro-ophthalmologist', 0.413), 
 ('retinal', 0.405), ('pistol-whipped', 0.400), ('vein/artery', 0.392)]
```
This model successfully identifies medically relevant terms:
- 'retinal' 
- 'neuro-ophthalmologist' 
- 'resorption' - medical process that can affect the retina
- 'vein/artery' 

#### Query 2: Medication - "ibuprofen"

**Better Model 2 Results:**
```
[('motrin', 0.599), ('advil', 0.574), ('400-600', 0.554), 
 ('celebrex', 0.545), ('nuprin', 0.526), ('aleve', 0.506), 
 ('tylenol', 0.502), ('acetaminophen', 0.498)]
```
The optimized model excellently captures medication relationships such as Brand names of ibuprofen (Motrin, Advil, Nuprin), Common dosage (400-600mg), Other NSAIDs (Celebrex, Aleve), Related pain medications (Tylenol, acetaminophen)

The optimized model (Better Model 2) learned medically meaningful relationships, identifying related anatomical structures, drug families, and clinical concepts. 
