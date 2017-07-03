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

import java.util.ArrayList;
import java.util.Arrays;
import bdl.activationFunctions.ActivationFunction;
import bdl.activationFunctions.SigmoidFunction;

/**
 *
 * @author chuckwolber
 */
public class Network
{
    private final ArrayList<Layer> _layers = new ArrayList<>();
    
    public static void main(String[] args) {
        mazur();
    }
    
    /**
     * In order to test this code, I utilized the weights, biases, and inputs
     * found in Matt Mazur's backpropagation example found here:
     * https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
     */
    public static void mazur() {
                SigmoidFunction sf = new SigmoidFunction();
        ArrayList<Double> inputs = new ArrayList<>(Arrays.asList(
                0.05, 0.10));
        ArrayList<Double> weights = new ArrayList<>(Arrays.asList(
                0.15, 0.25, 0.20, 0.30, 0.40, 0.50, 0.45, 0.55));
        ArrayList<Double> biases = new ArrayList<>(Arrays.asList(
                0.35, 0.35, 0.60, 0.60));
        ArrayList<Double> expected = new ArrayList<>(Arrays.asList(0.01, 0.99));
        Network nw = new Network(3, 2, sf, 0.5);
        nw.setInput(inputs);
        nw.setWeights(weights);
        nw.setBiases(biases);
        int i = 0;
        while (true) {
            nw.forwardPropagate();
            System.out.println(++i + " Output: " + nw.output());
            nw.calculateErrorRate(expected);
            nw.backwardPropagate();
            nw.updateWeights();
        }
    }
    
    public Network(int layers, int nodesPerLayer, ActivationFunction func, double learningRate) {
        for (int i=0; i<layers; i++)
            addLayer(new Layer(nodesPerLayer, func, learningRate));
    }
    
    public ArrayList<Double> weights() {
        ArrayList<Double> weights = new ArrayList<>();
        _layers.forEach((layer) -> {
            weights.addAll(layer.weights());
        });
        return weights;
    }
    
    public void setWeights(ArrayList<Double> weights) {
        _layers.forEach((layer) -> {
            layer.setWeights(weights);
        });
    }
    
    public void setBiases(ArrayList<Double> biases) {
        _layers.forEach((layer) -> {
            if (layer != inputLayer())
                layer.setBiases(biases);
        });
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
