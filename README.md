Team name: Tomi és Réka

Team members: Szűcs Tamás (BGDOMU), Boros Réka (AF6LRA)

Projekt neve: Képgenerálás diffúziós modellekkel

requierments.txt: Azokat a Python csomagokat tartalmazza, amelyek a kód futattásához szükségesek.

data.py: Ez a python kód egy PyTorch adatbetöltő osztályt definiál, amely képeket tölt be egy meglévő adathalmazból, majd fokozatosan zajt ad hozzájuk az idő függvényében. 
A zajosítás mértékét egy koszinusz-alapú ütemezés határozza meg, ahol az idő előrehaladtával az eredeti kép egyre inkább zajosabbá válik. 
A fő cél az, hogy szimulálja a diffúziós modellekben használt zajosítási folyamatot, ahol a modellnek meg kell tanulnia a zajos képből visszaállítani az eredetit.
Az osztály minden lekéréskor egy zajos képet, a hozzáadott zajt és az alkalmazott időlépést adja vissza.

u_net.py: Ebben a kódban egy PyTorch-alapú U-Net neurális hálózatot hoztunk létre. 
Ennek két változata található meg benne: az alap Unet és egy figyelmi mechanizmusokkal bővített változat, az AttentionUnet.

train.py: Ez a kód két tanító ciklust definiál egy PyTorch-alapú neurális hálózat számára. Az első, train_loop nevű függvény egy teljes adathalmazon végez tanítást.
A második függvény, a batch_train_loop, egyetlen batchen működik, ellenőrzi, hogy túl tud-e tanulni a modell rajta.

inference.py: A python kód egy diffúziós alapú képgeneráló rendszert valósít meg PyTorch segítségével. 
A célja, hogy véletlenszerű zajból kiindulva fokozatosan egyre tisztább képeket hozzon létre egy betanított neurális hálózat segítségével, 
így egy teljes diffúziós képgeneráló rendszert alkot, amely véletlenszerű zajból képes élethű képeket előállítani.

image_generation_pipeline.ipynb: Ez a notebook összefoglalja a modulokat, tartalmazza a tanítást és a képgenerálást is.
Az tanításhoz két adathalmazt használ: a CelebA és Flowers102 adatbázisokat. A kód különálló modelleket ment ki mind a CelebA, mind a Flowers102 adathalmazok alapján.

link_to_pretrained_models.txt: Tartalmazza a linket az előre tanított modelljeinkhez.

A kiértékeléshez szükséges modulok:
- image_generation_pipeline.ipynb
- inference.py
- u_net.py
- valamelyik pretrained model a link_to_pretrained_models.txt-ben található linkről, egymappában a notebookal
- A megfelelő modulok importálása és a Global Variables beállítása után az Inference szekció megfelelő cellájának futtatásával kiértékelhető a model


