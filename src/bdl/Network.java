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

package bdl;

import bdl.activationFunctions.SigmoidFunction;
import java.util.ArrayList;
import java.util.Arrays;

/**
 *
 * @author chuckwolber
 */
public class Network
{
    private final NetworkDescriptor _descriptor;
    private final ArrayList<Layer> _layers = new ArrayList<>();
    
    public static void main(String[] args) {
        mazur();
    }
    
    /**
     * In order to test this code, I utilized the weights, biases, and inputs
     * found in Matt Mazur's backpropagation example found here:
     * https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
     * 
     * Expected values:
     * Iteration 1:
     *  Weights: [0.15, 0.25, 0.2, 0.3, 0.4, 0.5, 0.45, 0.55]
     *  Output: [0.7513650695523157, 0.7729284653214625]
     * 
     * Iteration 2:
     *  Weights: [0.1497807161327628, 0.24975114363236958, 0.19956143226552567, 0.29950228726473915, 
     *            0.35891647971788465, 0.5113012702387375, 0.4086661860762334, 0.5613701211079891]
     *  Output: [0.7420881111907824, 0.7752849682944595]
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
        
        ArrayList<Double> inputs = new ArrayList<>(Arrays.asList(0.05, 0.10));
        ArrayList<Double> expected = new ArrayList<>(Arrays.asList(0.01, 0.99));
        
        Network nw = new Network(nd);
        nw.setInput(inputs);
        
        int i = 0;
        while (true) {
            nw.forwardPropagate();
            System.out.println(++i + " Output: " + nw.output() + " Error: " + nw.currentError(expected));
            //System.out.println(i + " Weights: " + nw.weights());
            nw.calculateErrorRate(expected);
            nw.backwardPropagate();
            nw.updateWeights();
        }
    }
    
    public Network(NetworkDescriptor descriptor) {
        _descriptor = descriptor;
        for (int i=0; i<_descriptor.layers()-1; i++)
            addLayer(new Layer(_descriptor.nodesPerLayer(), _descriptor.activationFunction(), 
                    _descriptor.learningRate()));
        addLayer(new Layer(_descriptor.outputNodes(), _descriptor.activationFunction(), 
                _descriptor.learningRate()));
        setWeights(descriptor.initialWeights());
        setBiases(descriptor.biases());
    }
    
    public ArrayList<Double> weights() {
        ArrayList<Double> weights = new ArrayList<>();
        _layers.forEach((layer) -> {
            weights.addAll(layer.weights());
        });
        return weights;
    }
    
    public void setInput(ArrayList<Double> inputValues) {
        inputLayer().setInput(inputValues);
    }
    
    public ArrayList<Double> output() {
        return outputLayer().output();
    }
    
    public ArrayList<Layer> layers() {
        return _layers;
    }
    
    public void forwardPropagate() {
        _layers.forEach((layer) -> {
            layer.forwardPropagate();
        });
    }
    
    public double currentError(ArrayList<Double> expectedValues) {
        return outputLayer().currentError(expectedValues);
    }
    
    public void calculateErrorRate(ArrayList<Double> expectedValues) {
        outputLayer().calculateErrorRate(expectedValues);
    }
    
    public void backwardPropagate() {
        for (int i=_layers.size()-1; i>=0; i--)
            _layers.get(i).backwardPropagate();
    }
    
    public void updateWeights() {
        _layers.forEach((layer) -> {
            layer.updateWeights();
        });
    }
    
    private void setWeights(ArrayList<Double> weights) {
        if (weights == null || weights.isEmpty())
            return;
        _layers.forEach((layer) -> {
            layer.setWeights(weights);
        });
    }
    
    private void setBiases(ArrayList<Double> biases) {
        if (biases == null || biases.isEmpty())
            return;
        _layers.forEach((layer) -> {
            if (layer != inputLayer())
                layer.setBiases(biases);
        });
    }
    
    private Layer inputLayer() {
        return _layers.get(0);
    }
    
    private Layer outputLayer() {
        return _layers.get(_layers.size()-1);
    }
    
    private void addLayer(Layer l) {
        if (_layers.size() > 0)
            l.linkToLayer(_layers.get(_layers.size()-1));
        _layers.add(l);
    }
}
