Team name: Tomi és Réka
Team members: Szűcs Tamás (BGDOMU), Boros Réka (AF6LRA)
Project description: Image generation with diffusion models
    A projekt célja, hogy két különböző diffúziós modellt (a Denoising Diffusion Probabilistic Modelt (DDPM) és a Denoising Diffusion Implicit Modelt (DDIM)) implementáljuk
    és betanítsuk, majd ezek segítségével valósághű képeket generáljunk két különböző adatkészleten: a CelebA és a Flowers102 adatokon.
    Az első cél a DDPM (Denoising Diffusion Probabilistic Model) megvalósítása. Ez a modell iteratív zajcsökkentési folyamatot alkalmaz, ahol egy kép zajjal kerül bevezetésre,
    és a modell fokozatosan tanulja meg, hogyan állítsa vissza az eredeti, zajmentes képet.
    A második cél a DDIM (Denoising Diffusion Implicit Model) implementálása, amely a DDPM egy optimalizált változata. A DDIM determinisztikus mintavételezést biztosít,
    így gyorsabb képgenerálást tesz lehetővé, miközben a generált képek minősége nem csökken.
Functions of the files in the repository:
    ....py: adatok betöltése
Related works:
    Az egyik cikkben, kiváló minőségű képgenerálást értek el diffúziós probabilisztikus modellekkel (DDPM) a CIFAR10 és a 256x256 LSUN adatkészleteken. A legjobb eredményeinket egy
    olyan súlyozott variációs határ alapján érték el, amely egy új kapcsolatot használ a diffúziós probabilisztikus modellek és a denoising score matching, valamint a
    Langevin-dinamika között.
    Egy másik cikkben bemutatják a denoising diffusion implicit modelleket (DDIM-eket), amelyek gyorsabb mintavételezést tesznek lehetővé ugyanazzal a tanítási folyamattal,
    mint a DDPM-ek. A DDIM-ek nem-Markov diffúziós folyamatokat használnak, amelyek gyorsabb generálást eredményeznek, és 10×-50× gyorsabban állítanak elő mintákat.
How to run it: 
    A CelebA és a Flower 102 adatkészletet egyes Python csomagok közvetlenül is támogatják, így nem szükséges manuálisan letölteni és kezelned az adatokat. Az egyik ilyen csomag
    a torchvision, amely a PyTorch része, és tartalmazza az adatkészlet kezeléséhez szükséges funkcionalitást.