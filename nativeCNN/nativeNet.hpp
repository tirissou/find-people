#include "layers/convolution_layer.hpp"
#include "layers/max_pooling_layer.hpp"
#include "layers/relu_layer.hpp"
#include "layers/dense_layer.hpp"
#include "layers/softmax_layer.hpp"
#include "layers/cross_entropy_loss_layer.hpp"
#include "utils/people.hpp"

#include <iostream>
#include <vector>
#include <cassert>
#include <armadillo>
#include <boost/test/unit_test.hpp>
#include <fstream>

#define DEBUG false

int main(int argc, char ** argv)
{
  PeopleData md("../data");

  std::vector<arma::cube> trainData = md.getTrainData();
  std::vector<arma::vec> trainLabels = md.getTrainLabels();

  std::vector<arma::cube> validationData = md.getValidationData();
  std::vector<arma::vec> validationLabels = md.getValidationLabels();

  assert(trainData.size() == trainLabels.size());
  assert(validationData.size() == validationLabels.size());

  std::vector<arma::cube> testData = md.getTestData();

/*#if DEBUG
  std::cout << DEBUG_PREFIX
      << "Training data size: " << trainData.size() << std::endl;
  std::cout << DEBUG_PREFIX
      << "Validation data size: " << validationData.size() << std::endl;
  std::cout << DEBUG_PREFIX
      << "Test data size: " << testData.size() << std::endl;
  std::cout << DEBUG_PREFIX << std::endl;
#endif
*/

  const size_t TRAIN_DATA_SIZE = trainData.size();
  const size_t VALIDATION_DATA_SIZE = validationData.size();
  const size_t TEST_DATA_SIZE = testData.size();
  const double LEARNING_RATE = 0.05;
  const size_t EPOCHS = 100;
  const size_t BATCH_SIZE = 32;
  const size_t NUM_BATCHES = TRAIN_DATA_SIZE / BATCH_SIZE;

  // Define the network layers
  ConvolutionLayer c1(
      28,
      28,
      1,
      5,
      5,
      1,
      1,
      6);
  // Output is 24 x 24 x 6

  ReLULayer r1(
      24,
      24,
      6);
  // Output is 24 x 24 x 6

  MaxPoolingLayer mp1(
      24,
      24,
      6,
      2,
      2,
      2,
      2);
  // Output is 12 x 12 x 6

  ConvolutionLayer c2(
      12,
      12,
      6,
      5,
      5,
      1,
      1,
      16);
  // Output is 8 x 8 x 16

  ReLULayer r2(
      8,
      8,
      16);
  // Output is 8 x 8 x 16

  MaxPoolingLayer mp2(
      8,
      8,
      16,
      2,
      2,
      2,
      2);
  // Output is 4 x 4 x 16

  DenseLayer d(
      4,
      4,
      16,
      10);
  // Output is a vector of size 10

  SoftmaxLayer s(10);
  // Output is a vector of size 10

  CrossEntropyLossLayer l(10);

  // Initialize armadillo structures to store intermediate outputs (Ie. outputs
  // of hidden layers)
  arma::cube conv1Out = arma::zeros(24, 24, 6);
  arma::cube relu1Out = arma::zeros(24, 24, 6);
  arma::cube mp1Out = arma::zeros(12, 12, 6);
  arma::cube conv2Out = arma::zeros(8, 8, 16);
  arma::cube relu2Out = arma::zeros(8, 8, 16);
  arma::cube mp2Out = arma::zeros(4, 4, 16);
  arma::vec denseOut = arma::zeros(10);
  arma::vec softOut = arma::zeros(10);

  // Initialize loss and cumulative loss. Cumulative loss totals loss over all
  // training examples in a minibatch.
  double loss = 0.0;
  double cumLoss = 0.0;


  for (size_t epoch = 0; epoch < EPOCHS; epoch++)
  {
#if DEBUG
    std::cout << DEBUG_PREFIX << std::endl;
    std::cout << DEBUG_PREFIX << "Epoch # " << epoch << std::endl;
#endif
    for (size_t batchIdx = 0; batchIdx < NUM_BATCHES; batchIdx++)
    {
      // Generate a random batch.
      arma::vec batch(BATCH_SIZE, arma::fill::randu);
      batch *= (TRAIN_DATA_SIZE - 1);

      for (size_t i = 0; i < BATCH_SIZE; i++)
      {
        // Forward pass
        c1.Forward(trainData[batch[i]], conv1Out);
        r1.Forward(conv1Out, relu1Out);
        mp1.Forward(relu1Out, mp1Out);
        c2.Forward(mp1Out, conv2Out);
        r2.Forward(conv2Out, relu2Out);
        mp2.Forward(r2Out, mp2Out);
        d.Forward(mp2Out, denseOut);
        denseOut /= 100;
        s.Forward(denseOut, softOut);

        // Compute the loss
        loss = l.Forward(softOut, trainLabels[batch[i]]);
        cumLoss += loss;

        // Backward pass
        l.Backward();
        arma::vec gradWrtPredictedDistribution =
            l.getGradientWrtPredictedDistribution();
        s.Backward(gradWrtPredictedDistribution);
        arma::vec gradWrtSIn = s.getGradientWrtInput();
        d.Backward(gradWrtSIn);
        arma::cube gradWrtDIn = d.getGradientWrtInput();
        mp2.Backward(gradWrtDIn);
        arma::cube gradWrtMP2In = mp2.getGradientWrtInput();
        r2.Backward(gradWrtMP2In);
        arma::cube gradWrtR2In = r2.getGradientWrtInput();
        c2.Backward(gradWrtR2In);
        arma::cube gradWrtC2In = c2.getGradientWrtInput();
        mp1.Backward(gradWrtC2In);
        arma::cube gradWrtMP1In = mp1.getGradientWrtInput();
        r1.Backward(gradWrtMP1In);
        arma::cube gradWrtR1In = r1.getGradientWrtInput();
        c1.Backward(gradWrtR1In);
        arma::cube gradWrtC1In = c1.getGradientWrtInput();
      }

      // Update params
      d.UpdateWeightsAndBiases(BATCH_SIZE, LEARNING_RATE);
      c1.UpdateFilterWeights(BATCH_SIZE, LEARNING_RATE);
      c2.UpdateFilterWeights(BATCH_SIZE, LEARNING_RATE);
    }

#if DEBUG
    // Output loss on training dataset after each epoch
    std::cout << DEBUG_PREFIX << std::endl;
    std::cout << DEBUG_PREFIX << "Training loss: "
        << cumLoss / (BATCH_SIZE * NUM_BATCHES) << std::endl;
#endif

    // Compute the training accuracy after epoch
    double correct = 0.0;
    for (size_t i = 0; i < TRAIN_DATA_SIZE; i++)
    {
      // Forward pass
      c1.Forward(trainData[i], c1Out);
      r1.Forward(c1Out, r1Out);
      mp1.Forward(r1Out, mp1Out);
      c2.Forward(mp1Out, c2Out);
      r2.Forward(c2Out, r2Out);
      mp2.Forward(r2Out, mp2Out);
      d.Forward(mp2Out, dOut);
      dOut /= 100;
      s.Forward(dOut, sOut);

      if (trainLabels[i].index_max() == sOut.index_max())
        correct += 1.0;
    }


    // Compute validation accuracy after epoch
    cumLoss = 0.0;
    correct = 0.0;
    for (size_t i = 0; i < VALIDATION_DATA_SIZE; i++)
    {
      // Forward pass
      c1.Forward(validationData[i], c1Out);
      r1.Forward(c1Out, r1Out);
      mp1.Forward(r1Out, mp1Out);
      c2.Forward(mp1Out, c2Out);
      r2.Forward(c2Out, r2Out);
      mp2.Forward(r2Out, mp2Out);
      d.Forward(mp2Out, dOut);
      dOut /= 100;
      s.Forward(dOut, sOut);

      cumLoss += l.Forward(sOut, validationLabels[i]);

      if (validationLabels[i].index_max() == sOut.index_max())
        correct += 1.0;
    }



    // Reset cumulative loss and correct count
    cumLoss = 0.0;
    correct = 0.0;

    // Write results on test data to results csv
    std::fstream fout("results_epoch_" + std::to_string(epoch) + ".csv",
                      std::ios::out);
    fout << "ImageId,Label" << std::endl;
    for (size_t i=0; i<TEST_DATA_SIZE; i++)
    {
      // Forward pass
      c1.Forward(testData[i], c1Out);
      r1.Forward(c1Out, r1Out);
      mp1.Forward(r1Out, mp1Out);
      c2.Forward(mp1Out, c2Out);
      r2.Forward(c2Out, r2Out);
      mp2.Forward(r2Out, mp2Out);
      d.Forward(mp2Out, dOut);
      dOut /= 100;
      s.Forward(dOut, sOut);

      fout << std::to_string(i+1) << ","
          << std::to_string(sOut.index_max()) << std::endl;
    }
    fout.close();
  }
}