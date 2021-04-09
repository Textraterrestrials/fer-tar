# Naš papir
Ovaj dokument sadrži sažetak ideja, misli, razmatranja, problema, ali i njihovih rješenja vezanih uz projekt u sklopu predmeta Analiza i pretraživanje teksta (engl. *Text Analysis and Retrieval*) u nadležnosti izv. prof. dr. sc. Jana Šnajdera. Naš je zadatak koji implementirati model koji je u stanju spoznati zdrav razum (engl. *common sense*), a zadan je kao četvrti zadatak na natjecanju SemEval koje se održalo 2020. godine. 

Zdrav razum nije usađen u temeljne principe svemira niti je konstantan, već se razvija tisućama godina. Stoga svaki čovjek mora naučiti zdrav razum i za to je nekima potrebno više vremena, a nekima manje. Vjerujem da nijedan čovjek ne može naučiti zdrav razum u samo dva mjeseca života bez obzira na to koliko stručnjaka mu u tome pomaže svojim znanjem. Međutim, može li to bijedni matematički model koji naprosto ulaznom vektoru pridružuje neku oznaku? Cilj ovog projekta je ponuditi potvrdan odgovor na ovo pitanje ali i bilo kakav odgovor na neka druga pitanja koja će se možda putem pojaviti. U tu svrhu nastao je ovaj dokument

## Pitanja na koja nastojimo odgovoriti (osim metapitanja navedenog u uvodu)
- Mogu li *data augmentation* tehnike zadržati ili čak poboljšati učenje zdravog razuma?
-  Koji su modeli najrobusniji na šum umjetno proizveden *data augmentation* tehnikama?

## Načela provođenja eksperimenata
1. Prilikom pisanja koda treba paziti na ponovnu iskoristivost koda, jedinstvenu odgovornost komponenti i ortogonalnost
2. Eventualni nusprodukti eksperimenata koji mogu biti korisni ostalim članovima tima ili za buduće eksperimente trebaju biti izdvojeni od koda vezanog uz sami eksperiment
3. Modele treba spremati u `models.py`
4. Zadatci su članovima tima pridruženi u `TODO.md`
5. U `MODELS.md` nalazi se checklista s raznim modelima uz resurse korištene ili potrebne za njihovu implementaciju
6. Podatke treba pohranjivati na Google Drive i to tako da svi članovi tima imaju pristup
7. Funkcije gubitka treba spremati u `losses.py`
8. Enkodere treba spremati u `encoders.py`
9. Korisne skripte treba spremati u `scripts.py`
10. Kad god je moguće treba koristiti `PyTorch` kako bi se izbjegla nekompatibilnost s ostalim komponentama
11.  Rezultate eksperimenata treba dokumentirati
12.  U slučaju nedostatka vremena ili druge više sile navedenih se načela ne treba držati pod svaku cijenu