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

/**
 *
 * @author chuckwolber
 */
public class Network
{
    private final NetworkDescriptor _descriptor;
    private final ArrayList<Layer> _layers = new ArrayList<>();
        
    public Network(NetworkDescriptor descriptor) {
        _descriptor = descriptor;
        for (int i=0; i<_descriptor.layers()-1; i++)
            addLayer(new Layer(_descriptor.nodesPerLayer(), 
                    _descriptor.activationFunction(), 
                    _descriptor.learningRate()));
        addLayer(new Layer(_descriptor.outputNodes(), 
                _descriptor.activationFunction(), 
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
    
    public void setInput(double[] inputValues) {
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
    
    public double currentError(double[] expectedValues) {
        return outputLayer().currentError(expectedValues);
    }
    
    public void calculateErrorRate(double[] expectedValues) {
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
