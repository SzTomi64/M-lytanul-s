Team name: Tomi és Réka

Team members: Szűcs Tamás (BGDOMU), Boros Réka (AF6LRA)

Project description: Image generation with diffusion models

A projekt célja, hogy két különböző diffúziós modellt (a Denoising Diffusion Probabilistic Modelt (DDPM) és a Denoising Diffusion Implicit Modelt (DDIM)) implementáljuk és betanítsuk, majd ezek segítségével valósághű képeket generáljunk két különböző adatkészleten: a CelebA és a Flowers102 adatokon.

Az első cél a DDPM (Denoising Diffusion Probabilistic Model) megvalósítása. Ez a modell iteratív zajcsökkentési folyamatot alkalmaz, ahol egy kép zajjal kerül bevezetésre, és a modell fokozatosan tanulja meg, hogyan állítsa vissza az eredeti, zajmentes képet.

A második cél a DDIM (Denoising Diffusion Implicit Model) implementálása, amely a DDPM egy optimalizált változata. A DDIM determinisztikus mintavételezést biztosít, így gyorsabb képgenerálást tesz lehetővé, miközben a generált képek minősége nem csökken.
    
Functions of the files in the repository:

image_generation.ipynb: Ez a nootbook tartalmazza az adatok letöltését és előkészítését, a modellt, valamit a tanító ciklust (jelenleg csak az adatok letöltését és előkészítését).
    
requierments.txt: Tartalmazza a szükséges python package-eket verziószámokkal, pip install -r requirements.txt command segítségével a megfelelő könyvtárak telepíthetőek.
    
Related works:

https://arxiv.org/pdf/2006.11239

https://arxiv.org/pdf/2105.05233

Az első cikkben, kiváló minőségű képgenerálást értek el diffúziós probabilisztikus modellekkel (DDPM) a CIFAR10 és a 256x256 LSUN adatkészleteken. A legjobb eredményeinket egy olyan súlyozott variációs határ alapján érték el, amely egy új kapcsolatot használ a diffúziós probabilisztikus modellek és a denoising score matching, valamint a Langevin-dinamika között.

A második cikkben bemutatják a denoising diffusion implicit modelleket (DDIM-eket), amelyek gyorsabb mintavételezést tesznek lehetővé ugyanazzal a tanítási folyamattal, mint a DDPM-ek. A DDIM-ek nem-Markov diffúziós folyamatokat használnak, amelyek gyorsabb generálást eredményeznek, és 10×-50× gyorsabban állítanak elő mintákat.


2. Mérföldkő:

Két ismert képadatbázist töltünk be a projekt számára: a CelebA és a Flowers102 nevű adatbázisokat. Ezek az adatbázisok a gépi tanulási modellek tanítására szolgáló, előre meghatározott képkészleteket tartalmazzák. Az először megadjuk a fájlok helyét, ahová majd letöltjük az adatbázisokat. Az elérési útvonalak a celebA_path és a Flowers102_path változókban vannak tárolva. Ez a két elérési útvonal meghatározza azt a helyet a gépen, ahová az egyes adatbázisok fájljai letöltődnek. Ezután a torchvision segítségével letöltjük az adatbázist a már előre meghatározott elérési útra. Ezek a kódok automatikusan elvégzik a letöltést, ha a fájlok még nincsenek a megadott elérési útvonalakon. Ezután a celebA és Flowers102 változókban a letöltött adathalmazok elérhetők, amelyeket a modell tanításához és kiértékeléséhez használhatunk.

Az eljárás során először egy adathalmaz betöltésével kezdünk, amelyen standard normális eloszlású zajt alkalmazunk. A zaj hozzáadását az időfüggvényében szabályozzuk, koszinusz görbével. Ennek során az egyes időszeletek (time step, T=500) mentén kis időértékeknél (t) kevesebb zajt adunk a képhez, míg nagyobb értékeknél több zaj kerül hozzáadásra. Ezáltal a kezdeti időszakban a kép részletgazdagságának jelentősebb része megmarad, míg későbbi időkben egyre kevésbé felismerhető a kép eredeti tartalma.

A modell tanítása a zaj előrejelzésére irányul, ahol a bemeneti változók a zajos kép, valamint az időpont (t), amely a pillanatnyi időpontot jelöli. A modell architektúrájaként egy U-net modellt választottunk, amely körülbelül 34 millió paraméterrel rendelkezik – bár a paraméterek száma adott esetben további növelést igényelhet a megfelelő teljesítmény eléréséhez. A modellt tovább gazdagítjuk egy szinuszos idő-beágyazási (time-embedding) technikával, amely során az időt szinusz- és koszinusz-függvények segítségével transzformáljuk, majd egy lineáris réteggel megfelelő méretűre alakítjuk, biztosítva az időbeli információ feldolgozását a modell különböző szintjein.

A veszteségfüggvényként L2 veszteséget (loss) alkalmazunk, amely a predikált és a tényleges zaj közötti négyzetes eltérést minimalizálja.

Az adathalmaz komplexitása miatt a modell jelenleg inkább alultanul (underfitting), vagyis nem képes minden mintázatot pontosan lekövetni. Ennek ellenére a fejlesztés során célunk, hogy a modell generált képei minél valósághűbbek legyenek. Az elért eredmények minőségét vizuálisan is értékeljük. Az optimális állapotot akkor tekintjük elérhetőnek, ha a generált képek szemre valósághűek. Jelenleg azonban a generált képek eltérnek a kívánt valósághűségtől, ami további finomhangolást igényel.
