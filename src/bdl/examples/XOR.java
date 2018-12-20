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
public class XOR
{
    public static void main(String[] args) {
        xor();
    }
    
    public static void xor() {
        NetworkDescriptor nd = new NetworkDescriptor();
        nd.setLayers(4);
        nd.setNodesPerLayer(2);
        nd.setOutputNodes(1);
        nd.setLearningRate(0.5);
        nd.setActivationFunction(new SigmoidFunction());
        
        double[] input1 = {0, 0};
        double[] input2 = {0, 1};
        double[] input3 = {1, 0};
        double[] input4 = {1, 1};
        double[] inputs[] = {input1, input2, input3, input4};
        double[] expected = {0, 1, 1, 0};
        
        Network nw = new Network(nd);
        double[] inpt;
        double expt;
        double error = 0;
        int i = 0;
        int epochMax = 400000;
        ArrayList<Double> initialWeights = nw.weights();
        while (i <= epochMax) {
            //System.out.println("Epoch: " + i);
            for (int x=0; x<inputs.length; x++) {
                inpt = inputs[x];
                expt = expected[x];
                
                nw.setInput(inpt);
                nw.forwardPropagate();
                if (i == epochMax)
                    System.out.println("\t" + Arrays.toString(inpt) + " " + nw.output());
                error += nw.currentError(new double[] {expt});
                nw.calculateErrorRate(new double[] {expt});
                nw.backwardPropagate();
                nw.updateWeights();
            }
            System.out.println("Epoch: " + i + " Error: " + error);
            i++;
            error = 0;
        }
        System.out.println("Initial Weights: " + initialWeights);
        System.out.println("Weights: " + nw.weights());
    }
}
