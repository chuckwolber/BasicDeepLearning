/**
 * Copyright 2017 Chuck Wolber
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy 
 * of this software and associated documentation files (the "Software"), to deal 
 * in the Software without restriction, including without limitation the rights 
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
 * copies of the Software, and to permit persons to whom the Software is 
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in 
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package bdl.examples;

import bdl.Network;
import bdl.NetworkDescriptor;
import bdl.activationFunctions.SigmoidFunction;
import java.util.ArrayList;
import java.util.Arrays;

/**
 *
 * @author chuckwolber
 */
public class Mazur
{
    public static void main(String[] args) {
        mazur();
    }
    
    /**
     * I utilized the weights, biases, and inputs found in Matt Mazur's 
     * backpropagation example found here:
     * 
     * https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
     * 
     * Expected values:
     * Iteration 1:
     *  Weights: [0.15, 0.25, 0.2, 0.3, 0.4, 0.5, 0.45, 0.55]
     *  Output:  [0.7513650695523157, 0.7729284653214625]
     *  Error:   [0.2983711087600027]
     * 
     * Iteration 2:
     *  Weights: [0.1497807161327628, 0.24975114363236958, 0.19956143226552567, 0.29950228726473915, 
     *            0.35891647971788465, 0.5113012702387375, 0.4086661860762334, 0.5613701211079891]
     *  Output:  [0.7420881111907824, 0.7752849682944595]
     *  Error:   [0.29102777369359933]
     */
    public static void mazur() {
        NetworkDescriptor nd = new NetworkDescriptor();
        nd.setLayers(3);
        nd.setNodesPerLayer(2);
        nd.setOutputNodes(2);
        nd.setLearningRate(0.5);
        nd.setBiases(new ArrayList<>(Arrays.asList(0.35, 0.35, 0.60, 0.60)));
        nd.setInitialWeights(new ArrayList<>(Arrays.asList(0.15, 0.25, 0.20, 0.30, 0.40, 0.50, 0.45, 0.55)));
        nd.setActivationFunction(new SigmoidFunction());
        
        double[] inputs = {0.05, 0.10};
        double[] expected = {0.01, 0.99};
        
        Network nw = new Network(nd);
        nw.setInput(inputs);
        
        int i = 0;
        while (i < 2) {
            nw.forwardPropagate();
            System.out.println(++i + " Output: " + nw.output() + " Error: " + nw.currentError(expected));
            System.out.println(i + " Weights: " + nw.weights());
            nw.calculateErrorRate(expected);
            nw.backwardPropagate();
            nw.updateWeights();
        }
    }
}
