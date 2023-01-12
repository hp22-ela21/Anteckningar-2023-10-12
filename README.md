# Anteckningar 2023-01-12
Implementering av enklare neuralt nätverk i C++ via klassen ann samt strukten dense_layer.

Filen "app.hpp" innehåller klassen ann, som utgörs av ett neuralt nätverk innehållande ett ingångslager, 
ett dolt lager samt ett utgångslager med godtyckligt antal noder i varje lager. 
Träningsdata passeras via referenser till vektorer.

Filen "dense_layer.hpp" innehåller strukten dense_layer, som används för implementeringen av dense-lager.

I filen "main.cpp" tränas ett neuralt nätverk bestående av två ingångar, två noder i det dolda lagret samt en utgång till att detektera ett 2-ingångars XOR-mönster. Träning sker under 1000 epoker med en lärhastighet på 2 %. 
Efter slutförd träning genomförs testning av nätverket via träningsdatan.