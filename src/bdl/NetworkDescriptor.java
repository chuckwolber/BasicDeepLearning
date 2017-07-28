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
public class NetworkDescriptor
{
    private int _layers;
    private int _nodesPerLayer;
    private int _outputNodes;
    private double _learningRate;
    ArrayList<Double> _biases;
    ArrayList<Double> _initialWeights;
    private ActivationFunction _function;
    
    public void setLayers(int layers) {
        _layers = layers;
    }
    
    public int layers() {
        return _layers;
    }
    
    public void setNodesPerLayer(int nodesPerLayer) {
        _nodesPerLayer = nodesPerLayer;
    }
    
    public int nodesPerLayer() {
        return _nodesPerLayer;
    }
    
    public void setOutputNodes(int outputNodes) {
        _outputNodes = outputNodes;
    }
    
    public int outputNodes() {
        if (_outputNodes == 0)
            return _nodesPerLayer;
        return _outputNodes;
    }
    
    public void setLearningRate(double learningRate) {
        _learningRate = learningRate;
    }
    
    public double learningRate() {
        return _learningRate;
    }
    
    public void setBiases(ArrayList<Double> biases) {
        _biases = biases;
    }
    
    public ArrayList<Double> biases() {
        return _biases;
    }
    
    public void setInitialWeights(ArrayList<Double> initialWeights) {
        _initialWeights = initialWeights;
    }
    
    public ArrayList<Double> initialWeights() {
        return _initialWeights;
    }
    
    public void setActivationFunction(ActivationFunction function) {
        _function = function;
    }
    
    public ActivationFunction activationFunction() {
        return _function;
    }
}
