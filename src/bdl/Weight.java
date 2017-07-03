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

/**
 *
 * @author chuckwolber
 */
public class Weight
{
    private double _weight;
    private Node _parentNode;
    private Node _childNode;
    private double _dEdw;
    private final Double _learningRate;
    
    public Weight(double learningRate) {
        _weight = RandomWeight.INSTANCE.nextDouble();
        _learningRate = learningRate;
    }
    
    public Node parentNode() {
        return _parentNode;
    }
    
    public void setParentNode(Node parentNode) {
        _parentNode = parentNode;
    }
    
    public Node childNode() {
        return _childNode;
    }
    
    public void setChildNode(Node childNode) {
        _childNode = childNode;
    }
    
    public double weight() {
        return _weight;
    }
    
    public void updateWeight() {
        _weight -= _dEdw * _learningRate;
    }
    
    public void setWeight(double weight) {
        _weight = weight;
    }
    
    public double dEdw() {
        return _dEdw;
    }
    
    public void setdEdw(double dEdw) {
        _dEdw = dEdw;
    }
}
