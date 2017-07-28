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
public class Node
{
    private final ArrayList<Weight> _parents = new ArrayList<>();
    private final ArrayList<Weight> _children = new ArrayList<>();
    private double _y = 0.0;
    private double _x = 0.0;
    private double _b = 0.0;
    private double _dEdy = 0.0;
    private final ActivationFunction _function;
    private final Double _learningRate;
    
    void sensorInput(double input) {
        _y = input;
    }
    
    double output() {
        return _y;
    }
    
    double x() {
        return _x;
    }
    
    double dEdy() {
        return this._dEdy;
    }
    
    Node(ActivationFunction function, Double learningRate) {
        _function = function;
        _learningRate = learningRate;
    }
    
    void setBias(double bias) {
        _b = bias;
    }
    
    ArrayList<Double> weights() {
        ArrayList<Double> weights = new ArrayList<>();
        _children.forEach((weight) -> {
            weights.add(weight.weight());
        });
        return weights;
    }
    
    void setWeights(ArrayList<Double> weights) {
        _children.forEach((weight) -> {
            weight.setWeight(weights.get(0));
            weights.remove(0);
        });
    }
    
    void linkToParent(Node parentNode) {
        Weight w = new Weight(_learningRate);
        w.setParentNode(parentNode);
        w.setChildNode(this);
        parentNode.addChildWeight(w);
        this.addParentWeight(w);
    }
    
    private void addParentWeight(Weight weight) {
        _parents.add(weight);
    }
    
    private void addChildWeight(Weight weight) {
        _children.add(weight);
    }
    
    private ActivationFunction getFunction() {
        return _function;
    }
    
    void forwardPropagate() {
        if (_parents.isEmpty())
            return;
        _x = 0.0;
        _parents.forEach((weight) -> {
            _x += weight.weight() * weight.parentNode().output();
        });
        _x += _b;
        _y = _function.evalFunction(_x);
    }
    
    double currentError(double expectedValue) {
        return (_y - expectedValue)*(_y - expectedValue);
    }
    
    void calculateErrorRate(double expectedValue) {
        _dEdy = _y - expectedValue;
    }
    
    /**
     * The Backpropagation algorithm calculates how the overall error changes 
     * when you change a given weight in a deep learning network. In simple 
     * English this is "the rate of error change with respect to the value of 
     * the weight". 
     * 
     * For those uninitiated in the mathematics, this is similar to "miles
     * per hour", where you would say "the change in miles with respect to the
     * number of hours". Saying "error per weight" is a bit weird, but if that
     * makes it as easy to understand as "miles per hour", then feel free.
     * 
     * The Backpropagation formula is as follows:
     * 
     * ∂E/∂w = (∂x/∂w) * (∂y/∂x) * (∂E/∂y)
     * 
     * Where: ∂x/∂w = y
     * Where: ∂y/∂x = f'(x), where f(x) is the Activation Function.
     * Where: ∂E/∂y = Sum of: (∂xc/∂y) * (∂yc/∂xc) * (∂E/∂yc)
     * Where: ∂xc/∂y = wc, and wc = the child node's weight.
     * Where: ∂yc/∂xc = f'(xc), where f(xc) is the Activation Function. 
     * Where: ∂E/∂yc = y - expectedValue
     * Where: ∂ is the partial derivative symbol.
     * Where: c refers to the child node.
     */
    void backwardPropagate() {
        if (!_children.isEmpty())
            _dEdy = 0.0;
        _children.forEach((weight) -> {
            double x = weight.childNode().x();
            _dEdy += weight.weight() *
                         weight.childNode().getFunction().evalFunctionDerivative(x) *
                         weight.childNode().dEdy();
        });
        _parents.forEach((weight) -> {
           weight.setdEdw(weight.parentNode().output() *
                   weight.childNode().getFunction().evalFunctionDerivative(_x) *
                   weight.childNode().dEdy());
        });
    }
    
    void updateWeights() {
        _children.forEach((weight) -> {
            weight.updateWeight();
        });
    }
}
