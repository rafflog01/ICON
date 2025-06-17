from owlready2 import *
import os
import pandas as pd

print("ONTOLOGIA\n")

# Ottieni il percorso assoluto del file
current_dir = os.path.dirname(os.path.abspath(__file__))
owl_file = os.path.join(current_dir, "breast_ontology.owl")

onto = get_ontology(owl_file).load()

# Stampa di tutte le classi dell'ontologia
print("####################################################################################################")
print("LISTA DELLE CLASSI DELL'ONTOLOGIA\n")
classes = list(onto.classes())
for cls in classes:
    print(f"• CLASSE: {cls.name}")
print()

# Stampa di tutte le object properties dell'ontologia
print("####################################################################################################")
print("OBJECT PROPERTIES DELL'ONTOLOGIA\n")
object_properties = list(onto.object_properties())
for prop in object_properties:
    print(f"• PROPRIETÀ: {prop.name}")
print()

# Stampa di tutte le data properties dell'ontologia
print("####################################################################################################")
print("DATA PROPERTIES DELL'ONTOLOGIA\n")
data_properties = list(onto.data_properties())
for prop in data_properties:
    print(f"• PROPRIETÀ: {prop.name}")
print()

print("####################################################################################################")
print("Lista Person nella ontologia:\n")
persons = onto.search(is_a=onto.Person)
print([p.name for p in persons], "\n")

print("Lista Cancer nella ontologia:\n")
cancers = onto.search(is_a=onto.Cancer)
print([c.name for c in cancers], "\n")

print("Lista Breast_cancer nella ontologia:\n")
b_cancers = onto.search(is_a=onto.Breast_cancer)
print([bc.name for bc in b_cancers], "\n")

print("Lista Analysis nella ontologia:\n")
analysis = onto.search(is_a=onto.Analysis)
print([a.name for a in analysis], "\n")

print("Lista di persone che hanno un cancro al seno:\n")
patients = onto.search(is_a=onto.Person, has_breast_cancer="*")
print([p.name for p in patients], "\n")
