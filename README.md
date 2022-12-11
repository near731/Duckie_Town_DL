## Deep Learning a gyakorlatban Python és Lua alapon (BMEVITMAV45)

## Nagy Házi feladat: Önvezető autózás Duckietown környezetben

# Adatok:

## "Csapat" adatok:

Név: DLuckies

Tag(ok) neve:
- Németh Áron Imre (D1J5ZG)

## Felhasznált szoftverek verziószámai

- OS: Ubuntu 22.04
- Python: 3.8.15
- Tensorflow: 2.11.0

# Használati útmutató:

Először klónozzuk a gym-duckietown repot!

```
git clone https://github.com/duckietown/gym-duckietown.git
cd gym-duckietown
```
Majd pedig ezt a repot.

```
https://github.com/near731/Duckie_Town_DL.git
cd Duckie_Town_DL
```

## CONV model használata

Adatok generálása manuális szimulációval:

```
python3.8 CONV_imitation.py
```

Tanítás:

```
python3.8 CONV_train.py 
```
Végül a tesztelés:

```
python3.8 CONV_test.py 
```
Az adatokat ide nem tudtam a túl nagy mérete miatt feltölteni, ezért Google Drive-ra töltöttem fel:

Link: https://drive.google.com/drive/folders/17UHHcnYKoU3VDA1TEb9__AykRrXvCtXq

Ezt a .zip file-t a Duckie_Town_DL mappába kell kicsomagolni.

## Q model használata

Tanítás:

```
python3.8 Q_train.py 
```
Tesztelés:

```
python3.8 Q_test.py 
```



# Egyéb

Sajnos a Linux szükségessége miatt 2 csapattárs is kihátrált a projekt mögül, a kezdeti sikertelenségek után visszaléptek a feladattól. Végül egyedül megpróbáltam, de a feladat nagyon nehéznek bizonyult számomra, valamint a tanításra és a tesztelésre már nagyon kevés időm maradt. 
