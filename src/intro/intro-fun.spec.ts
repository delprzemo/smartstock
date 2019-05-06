import { expect } from 'chai';
import * as brain from 'brain.js'

describe('Intro - fun', () => {

    const commonLSTMOptions = {
        iterations: 200000,    // the maximum times to iterate the training data --> number greater than 0
        errorThresh: 0.05,   // the acceptable error percentage from training data --> number between 0 and 1
        log: false,           // true to use console.log, when a function is supplied it is used --> Either true or a function
        logPeriod: 10,        // iterations between logging out --> number greater than 0
        learningRate: 0.3,    // scales with delta to effect training rate --> number between 0 and 1
        momentum: 0.1,        // scales with next layer's change value --> number between 0 and 1
        callbackPeriod: 10,   // the number of iterations through the training data between callback calls --> number greater than 0
        timeout: Infinity     // the max number of milliseconds to train for --> number greater than 0
    }

    it("Happy/Sad example", () => {
        // const net = new brain.recurrent.LSTM();

        // net.train([
        //     { input: 'I feel great about the world!', output: 'happy' },
        //     { input: 'The world is a terrible place!', output: 'sad' },
        // ], commonLSTMOptions);

        // const output = net.run('I feel great about the world!');  // 'happy'
    })


    // it('First test with brain.js LSTMT', () => {
    //     const net = new brain.recurrent.LSTM();


    //     net.train([
    //         { input: [1, 1], output: [3] },
    //         { input: [2, 2], output: [6] },
    //         { input: [3, 1], output: [7] },
    //         { input: [6, 1], output: [13] },
    //         { input: [9, 8], output: [26] },
    //         { input: [-10, 2], output: [-18] },
    //         { input: [-20, 20], output: [-20] },
    //         { input: [11, 22], output: [44] },
    //         { input: [111, -900], output: [-678] },
    //         { input: [1, 2], output: [4] },
    //         { input: [2, 1], output: [5] },
    //         { input: [2, 1], output: [5] },
    //         { input: [2, 3], output: [7] },
    //         { input: [213, 32], output: [458] },
    //         { input: [32424, -2], output: [64846] },
    //         { input: [-23, -234], output: [-280] },
    //         { input: [-234, -324], output: [-792] }
    //     ], commonLSTMOptions);

    //     const output = net.run([1, 3]);

    //     expect(output).to.equal(1);
    // });

    function generateSinArray(arrLength: number){
        let result = [];
        for(let i = 0; i < arrLength; i++) {
            result.push(Math.sin(i));
        }

        return result;
    }

    function normalize(arr: number[]) {
        var positiveRatio = Math.max(...arr);
        var negativeRatio = Math.min(...arr);
        var ratio = Math.abs(positiveRatio) >  Math.abs(negativeRatio) 
            ? Math.abs(positiveRatio) : Math.abs(negativeRatio);

        console.log(ratio);

        return arr.map(v => v / ratio);
    }

    function converToInputOutput(arr: number[], bucketLength: number): object {
        let result = [];
     
        for(let i = bucketLength; i < arr.length; i++) {
            let bucketArray = [];
            for(let bucketIndex = i - bucketLength; bucketIndex < i; bucketIndex++) {
                bucketArray.push(arr[bucketIndex]);
            }

            result.push({input: bucketArray, output: [arr[i]]});
        }

        return result;
    }

    it('First test with brain.js LSTMT', () => {

        const myCommonLSTMOptions = {
            iterations: 500000,    // the maximum times to iterate the training data --> number greater than 0
            errorThresh: 0.05,   // the acceptable error percentage from training data --> number between 0 and 1
            log: true,           // true to use console.log, when a function is supplied it is used --> Either true or a function
            logPeriod: 10,        // iterations between logging out --> number greater than 0
            learningRate: 0.3,    // scales with delta to effect training rate --> number between 0 and 1
            momentum: 0.1,        // scales with next layer's change value --> number between 0 and 1
            callbackPeriod: 10,   // the number of iterations through the training data between callback calls --> number greater than 0
            timeout: Infinity,     // the max number of milliseconds to train for --> number greater than 0
            callback: function() {
                
            }
        }

        const config = {
            inputSize: 1,
            hiddenLayers: [10],
            outputSize: 1
          };

        const net = new brain.recurrent.LSTMTimeStep(config);

        net.train([
            generateSinArray(100)
        ],
        myCommonLSTMOptions);

        let output = net.run([-0.916509490200547, 
            -0.158592906028573, 
            0.745133264557413, 
            0.963787348067422, 
            0.296339788497322, 
            -0.643561205976262
            ]);  // 3
    })


    // it('First test with brain.js Neural', () => {

    //     const myCommonLSTMOptions = {
    //         iterations: 200000,    // the maximum times to iterate the training data --> number greater than 0
    //         errorThresh: 0.01,   // the acceptable error percentage from training data --> number between 0 and 1
    //         log: true,           // true to use console.log, when a function is supplied it is used --> Either true or a function
    //         logPeriod: 10,        // iterations between logging out --> number greater than 0
    //         learningRate: 0.3,    // scales with delta to effect training rate --> number between 0 and 1
    //         momentum: 0.1,        // scales with next layer's change value --> number between 0 and 1
    //         callbackPeriod: 10,   // the number of iterations through the training data between callback calls --> number greater than 0
    //         timeout: Infinity,     // the max number of milliseconds to train for --> number greater than 0
    //         callback: function(val: any) {
    //         }
    //     }

    //     const net = new brain.NeuralNetwork;
        
    //     let trainData = converToInputOutput(generateSinArray(100),
    //         5);

    //     net.train(trainData, 
    //         myCommonLSTMOptions);

    //     let output = net.run([
    //         -0.158592906028573, 
    //         0.745133264557413, 
    //         0.963787348067422, 
    //         0.296339788497322, 
    //         -0.643561205976262
    //         ]);  // 3
    // })
});