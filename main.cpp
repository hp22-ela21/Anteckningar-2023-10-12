/********************************************************************************
* main.cpp: Implementering av ett enkelt neuralt nätverk i C++.
*           Slutför medlemsfunktionerna för klassen ann så att modellen
*           predikterar korrekt i enlighet med mönstret beskrivet nedan.
********************************************************************************/
#include "ann.hpp"
#include <vector>

/********************************************************************************
* main: Skapar ett neuralt nätverk innehållande två ingångsnoder, två noder
*       i det enda ingångslagret och två utgångsnoder. Nätverk tränas för att 
*       detektera ett 2-ingångars XOR-mönster under 1000 epoker med en 
*       lärhastighet på 2 %. Efter träningen genomförs prediktion samt utskrift
*       med samtliga träningsuppsättningars insignaler som indata.
* 
*       XOR-mönstret för insignaler X1 och X2 samt utsignal Y visas nedan:
*     
*       X1 X2 Y
*       0  0  0
*       0  1  1
*       1  0  1
*       1  1  0
********************************************************************************/
int main(void)
{
   const std::vector<std::vector<double>> train_in = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
   const std::vector<std::vector<double>> train_out = { { 0 }, { 1 }, { 1 }, { 0 } };
   
   ann ann1(2, 2, 1);
   ann1.set_training_data(train_in, train_out);
   ann1.train(1000, 0.02);
   ann1.print();

   return 0;
}