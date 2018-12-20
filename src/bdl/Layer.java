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

import bdl.activationFunctions.ActivationFunction;
import java.util.ArrayList;

/**
 *
 * @author chuckwolber
 */
public class Layer
{
    private final ArrayList<Node> _nodes = new ArrayList<>();
    
    Layer(int nodes, ActivationFunction func, double learningRate) {
        for (int i=0; i<nodes; i++)
            _nodes.add(new Node(func, learningRate));
    }
    
    ArrayList<Node> nodes() {
        return _nodes;
    }
    
    void setBiases(ArrayList<Double> biases) {
        _nodes.forEach((node) -> {
            node.setBias(biases.get(0));
            biases.remove(0);
        });
    }
    
    ArrayList<Double> weights() {
        ArrayList<Double> weights = new ArrayList<>();
        _nodes.forEach((node) -> {
            weights.addAll(node.weights());
        });
        return weights;
    }
    
    void setWeights(ArrayList<Double> weights) {
        _nodes.forEach((node) -> {
            node.setWeights(weights);
        });
    }
    
    void setInput(double[] inputValues) {
        if (inputValues.length != _nodes.size())
            return;
        for (int i=0; i<_nodes.size(); i++)
            _nodes.get(i).sensorInput(inputValues[i]);
    }
    
    ArrayList<Double> output() {
        ArrayList<Double> output = new ArrayList<>();
        _nodes.forEach((node) -> {
            output.add(node.output());
        });
        return output;
    }
    
    void forwardPropagate() {
        _nodes.forEach((node) -> {
            node.forwardPropagate();
        });
    }
    
    double currentError(double[] expectedValues) {
        if (expectedValues.length != _nodes.size())
            return -1.0;
        double currentError = 0.0;
        for (int i=0; i<_nodes.size(); i++)
            currentError += _nodes.get(i).currentError(expectedValues[i]);
        return 0.5*currentError;
    }
    
    void calculateErrorRate(double[] expectedValues) {
        if (expectedValues.length != _nodes.size())
            return;
        for (int i=0; i<_nodes.size(); i++)
            _nodes.get(i).calculateErrorRate(expectedValues[i]);
    }
    
    void backwardPropagate() {
        _nodes.forEach((node) -> {
            node.backwardPropagate();
        });
    }
    
    void updateWeights() {
        _nodes.forEach((node) -> {
            node.updateWeights();
        });
    }
    
    void linkToLayer(Layer parentLayer) {
        parentLayer.nodes().forEach((parentNode)-> {
            _nodes.forEach((node)-> {
                node.linkToParent(parentNode);
            });
        });
    }
}
