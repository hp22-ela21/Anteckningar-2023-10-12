/********************************************************************************
* dense_layer.hpp: Innehåller funktionalitet för implementering av dense-lager
*                  i neurala nätverk via strukten dense_layer.
********************************************************************************/
#ifndef DENSE_LAYER_HPP_
#define DENSE_LAYER_HPP_

/* Inkluderingsdirektiv: */
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdlib>

/********************************************************************************
* dense_layer: Strukt för implementering av dense-lager med valbart antal noder
*              samt vikter per nod i neurala nätverk. Bias och vikter för
*              samtliga noder erhåller randomiserade startvärden mellan 0 - 1,
*              övriga parametrar sätts till 0 vid start.
********************************************************************************/
struct dense_layer
{
   std::vector<double> output;               /* Nodernas utsignaler. */
   std::vector<double> error;                /* Nodernas uppmätta fel/avvikelser. */
   std::vector<double> bias;                 /* Nodernas vilovärden (m-värden). */
   std::vector<std::vector<double>> weights; /* Nodernas vikter (k-värden). */

   /********************************************************************************
   * dense_layer: Initierar nytt tomt dense-lager.
   ********************************************************************************/
   dense_layer(void) { }

   /********************************************************************************
   * dense_layer: Initierar nytt dense-lager av angiven storlek.
   * 
   *              - num_nodes  : Antalet noder i det nya dense-lagret.
   *              - num_weights: Antalet vikter per nod i det nya dense-lagret.
   ********************************************************************************/
   dense_layer(const std::size_t num_nodes,
               const std::size_t num_weights)
   {
      this->resize(num_nodes, num_weights);
      return;
   }

   /********************************************************************************
   * ~dense_layer: Tömmer angivet dense-lager om detta lager går ur scope,
   *               exempelvis om ett lokalt dense-lager har deklarerats och
   *               funktionen där detta lager har deklarerats avslutas (lagret
   *               frigörs då automatiskt från stacken).
   ********************************************************************************/
   ~dense_layer(void)
   {
      this->clear();
      return;
   }

   /********************************************************************************
   * num_nodes: Returnerar antalet befintliga noder i angivet dense-lager.
   ********************************************************************************/
   inline std::size_t num_nodes(void) const
   {
      return this->output.size();
   }

   /********************************************************************************
   * num_weights: Returnerar antalet vikter per nod i angivet dense-lager.
   ********************************************************************************/
   inline std::size_t num_weights(void) const
   {
      return this->weights.size() ? this->weights[0].size() : 0;
   }

   /********************************************************************************
   * clear: Tömmer angiven vektor.
   ********************************************************************************/
   void clear(void)
   {
      this->output.clear();
      this->error.clear();
      this->bias.clear();
      this->weights.clear();
      return;
   }

   /********************************************************************************
   * resize: Sätter antalet noder och vikter per nod i angiven vektor. Bias och
   *         vikter tilldelas randomiserade startvärden mellan 0 - 1, övriga
   *         parametrar sätts till 0 vid start. Innan storleken sätts så töms
   *         dense-lagret för att göra initieringen enklare.
   * 
   *         - num_nodes  : Antalet noder i dense-lagret.
   *         - num_weights: Antalet vikter per nod i dense-lagret.
   ********************************************************************************/
   void resize(const std::size_t num_nodes,
               const std::size_t num_weights)
   {
      this->output.resize(num_nodes, 0.0);
      this->error.resize(num_nodes, 0.0);
      this->bias.resize(num_nodes, 0.0);
      this->weights.resize(num_nodes, std::vector<double>(num_weights, 0.0));

      for (std::size_t i = 0; i < num_nodes; ++i)
      {
         this->bias[i] = this->get_random();

         for (std::size_t j = 0; j < num_weights; ++j)
         {
            this->weights[i][j] = this->get_random();
         }
      }

      return;
   }

   /********************************************************************************
   * print: Skriver ut flyttal från angiven vektor på en rad via angiven utström.
   *
   *        - data        : Referens till vektorn vars innehåll ska skrivas ut.
   *        - ostream     : Referens till angiven utström.
   *        - num_decimals: Antalet decimaler i utskriften.
   ********************************************************************************/
   static void print(const std::vector<double>& data,
                     std::ostream& ostream = std::cout,
                     const std::size_t num_decimals = 1)
   {
      ostream << std::fixed;

      for (auto& i : data)
      {
         ostream << std::setprecision(num_decimals) << i << " ";
      }

      ostream << "\n";
      return;
   }

   /********************************************************************************
   * print: Skriver ut information om angivet dense-lager via angiven utström,
   *        där standardutenheten std::cout används som default för utskrift i 
   *        terminalen.
   * 
   *        - ostream: Referens till angiven utström (default = std::cout).
   ********************************************************************************/
   void print(std::ostream& ostream = std::cout) const
   {
      ostream << "--------------------------------------------------------------------------------\n";
      ostream << "Number of nodes: " << this->num_nodes() << "\n";
      ostream << "Number of weights per node: " << this->num_weights() << "\n\n";

      ostream << "Output: ";
      this->print(this->output, ostream);

      ostream << "Error: ";
      this->print(this->error, ostream);

      ostream << "Bias: ";
      this->print(this->bias, ostream);

      ostream << "\nWeights:\n";

      for (std::size_t i = 0; i < this->num_nodes(); ++i)
      {
         ostream << "Node " << i + 1 << ": ";
         this->print(this->weights[i], ostream);
      }

      ostream << "--------------------------------------------------------------------------------\n\n";
      return;
   }

   /********************************************************************************
   * feedforward: Beräknar nya utsignaler för varje nod i angivet dense-lager
   *              genom att summera respektive nods bias samt indata (vikter *
   *              nya insignaler). Om denna summa överstiger 0 så är noden 
   *              aktiverad och utsignalen sätts till beräknad summa. Annars är 
   *              noden inaktiverad och utsignalen sätts då till 0.
   * 
   *              - input: Referens till vektor med nya insignaler.
   ********************************************************************************/
   void feedforward(const std::vector<double>& input)
   {
      for (std::size_t i = 0; i < this->num_nodes(); ++i)
      {
         auto sum = this->bias[i];

         for (std::size_t j = 0; j < this->num_weights() && j < input.size(); ++j)
         {
            sum += input[j] * this->weights[i][j];
         }

         this->output[i] = this->relu(sum);
      }

      return;
   }

   /********************************************************************************
   * backpropagate: Beräknar fel/avvikelser i angivet utgångslager via angivna
   *                referensvärden från träningsdatan. OBS! Denna medlemsfunktion
   *                är avsedd enbart för utgångslager, se den alternativa
   *                medlemsfunktionen med samma namn för dolda lager.
   * 
   *                - reference: Referens till vektor innehållande referensvärden.
   ********************************************************************************/
   void backpropagate(const std::vector<double>& reference)
   {
      for (std::size_t i = 0; i < this->num_nodes(); ++i)
      {
         const auto dev = reference[i] - this->output[i];
         this->error[i] = dev * delta_relu(this->output[i]);
      }

      return;
   }

   /********************************************************************************
   * backpropagate: Beräknar fel/avvikelser i angivet dolt lager via parametrar
   *                från nästa/efterföljande lager. OBS! Denna medlemsfunktion
   *                är avsedd enbart för dolda lager, se den alternativa
   *                medlemsfunktionen med samma namn för utgångslager.
   *
   *                - next_layer: Referens till nästa/efterföljande dense-lager.
   ********************************************************************************/
   void backpropagate(const dense_layer& next_layer)
   {
      for (std::size_t i = 0; i < this->num_nodes(); ++i)
      {
         auto sum = 0.0;

         for (std::size_t j = 0; j < next_layer.num_nodes(); ++j)
         {
            sum += next_layer.error[j] * next_layer.weights[j][i];
         }

         this->error[i] = sum * this->delta_relu(this->output[i]);
      }

      return;
   }

   /********************************************************************************
   * optimize: Justerar bias och vikter i angivet dense-lager utefter beräknade
   *           felvärden samt angiven lärhastighet. För att justera vikterna tas
   *           insignalerna i åtanke, då högre insignal innebär att en given vikt
   *           haft högre påverkan och därmed högre bidrag vid eventuellt fel.
   *           Därmed justeras vikter med höga insignaler i högre grad för att
   *           minska aktuella felvärden.
   * 
   *           - input        : Referens till vektor innehållande insignaler, 
   *                            som används för att justera vikterna.
   *           - learning_rate: Indikerar hur hög andel av aktuell fel som
   *                            bias och vikter ska justeras.
   ********************************************************************************/
   void optimize(const std::vector<double>& input,
                 const double learning_rate)
   {
      for (std::size_t i = 0; i < this->num_nodes(); ++i)
      {
         this->bias[i] += this->error[i] * learning_rate;

         for (std::size_t j = 0; j < this->num_weights() && j < input.size(); ++j)
         {
            this->weights[i][j] += this->error[i] * learning_rate * input[j];
         }
      }

      return;
   }

private:
   /********************************************************************************
   * get_random: Returnerar ett randomiserat flyttal mellan 0.0 - 1.0.
   ********************************************************************************/
   static inline double get_random(void)
   {
      return static_cast<double>(std::rand()) / RAND_MAX;
   }

   /********************************************************************************
   * relu: Implementerar en ReLU-funktion, där utsignalen y sätts till angiven
   *       summa sum om denna summa överstiger 0, annars sätts utsignalen till 0.
   *       Utsignalen y returneras efter beräkning. Om summan överstiger 0 så är 
   *       aktuell nod aktiverad, annars är den inaktiverad:
   * 
   *       sum > 0  => y = sum (noden aktiverad)
   *       sum <= 0 => y = 0   (noden inaktiverad)
   * 
   *       - sum: Summan av nodens bias samt  indata (insignaler * vikter).
   ********************************************************************************/
   static inline double relu(const double sum)
   {
      return sum > 0.0 ? sum : 0.0;
   }

   /********************************************************************************
   * delta_relu: Implementerar en funktion för att beräkna derivatan av 
   *             ReLU-funktionen, där utsignalen dy sätts till 1.0 om angiven nod
   *             är aktiverad, vilket kontrolleras via ingående argument output,
   *             som utgörs av nodens aktuella utsignal. Annars returneras 0.0.
   *             Via detta värde räknas endast fel/avvikelser på aktiverade noder,
   *             vilket också medför att enbart aktiverade noder justeras vid
   *             optimering. Utsignal dy returneras vid återhopp enligt nedan:
   *
   *             output > 0  => dy = 1 (noden aktiverad, fel beräknas)
   *             output <= 0 => dy = 0 (noden inaktiverad, fel beräknas inte)
   ********************************************************************************/
   static inline double delta_relu(const double output)
   {
      return output > 0.0 ? 1.0 : 0.0;
   }
};

#endif /* DENSE_LAYER_HPP_ */