Team name: Tomi és Réka

Team members: Szűcs Tamás (BGDOMU), Boros Réka (AF6LRA)

Project description: Image generation with diffusion models

data.py: Ez a python kód egy PyTorch adatbetöltő osztályt definiál, amely képeket tölt be egy meglévő adathalmazból, majd fokozatosan zajt ad hozzájuk az idő függvényében. 
A zajosítás mértékét egy koszinusz-alapú ütemezés határozza meg, ahol az idő előrehaladtával az eredeti kép egyre inkább zajosabbá válik. 
A fő cél az, hogy szimulálja a diffúziós modellekben használt zajosítási folyamatot, ahol a modellnek meg kell tanulnia a zajos képből visszaállítani az eredetit.
Az osztály minden lekéréskor egy zajos képet, a hozzáadott zajt és az alkalmazott időlépést adja vissza.

u_net.py: Ebben a kódban egy PyTorch-alapú UNet neurális hálózatot hoztunk létre. 
Ennek két változata található meg benne: az alap UNet és egy figyelmi mechanizmusokkal bővített változat, az Attention UNet.

train.py: Ez a kód két tanító ciklust definiál egy PyTorch-alapú neurális hálózat számára. Az első, train_loop nevű függvény egy teljes adathalmazon végez tanítást.
A második függvény, a batch_train_loop, egyetlen előre definiált adathalmaz-csomagon működik.

inference.py: A python kód egy diffúziós alapú képgeneráló rendszert valósít meg PyTorch segítségével. 
A célja, hogy véletlenszerű zajból kiindulva fokozatosan egyre tisztább képeket hozzon létre egy betanított neurális hálózat segítségével, 
így egy teljes diffúziós képgeneráló rendszert alkot, amely véletlenszerű zajból képes élethű képeket előállítani.

image_generation_pipeline.ipynb: Ebben a notebook-ban egy képregeneráló modellt tanítunk, amely zajos bemenetekből megtanulja visszaállítani az eredeti képeket.
Az tanításhoz két adathalmazt használ: a CelebA és Flowers102 adatbázisokat. A kód különálló modelleket ment ki mind a CelebA, mind a Flowers102 adathalmazok alapján.
