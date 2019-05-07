import { expect } from 'chai';
import * as tf from '@tensorflow/tfjs-node'

describe('Intro -tensorflow -  fun', () => {
    before(function () {
        this.timeout(1000000) // 10 second timeout for setup
    })

    // it("Basic trial", () => {

    //     // Tensor manipulation
    //     const t1 = tf.tensor(([1,2,3,4,2,4,6,8]), [2,4]);
    //     const squared = t1.square();


    // })

    // it("Simple prediction Y = 2X - 1", async function (done) {
    //     this.timeout(500000); // This works

    //     // declare models
    //     const model = tf.sequential(); //sequential model is any model where the outputs of one layer are the inputs to the next layer
    //     model.add(tf.layers.dense({units: 1, inputShape: [1]}));

    //     // Specify loss and optimizer for model
    //     model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    //     // Prepare training data
    //     const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4, 10], [7, 1]);
    //     const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7, 19], [7, 1]);


    //     // Train the model
    //     await model.fit(xs, ys, {epochs: 1000}).then(async () => {
    //         // Use model to predict values
    //         const result = await model.predict(tf.tensor2d([20], [1,1])).toString();
    //         done();
    //     });
    // })

    function generateSinTrainingData(arrLength: number) {
        let input = [];
        let output = [];
        for (let i = 0; i < arrLength; i++) {
            let rand = Math.random();
            input.push(rand);
            output.push(Math.sin(rand));
        }

        const xs = tf.tensor2d(input, [input.length, 1]);
        const ys = tf.tensor2d(output, [output.length, 1]);

        return [xs, ys];
    }

    function generateSinTimeSeries(arrLength: number) {
        let result = [];
        for (let i = 0; i < arrLength; i++) {
            result.push(Math.sin(i));
        }

        return result;
    }

    function generateTimeSeriesTensor(array: number[], windowSize: number) {
        let input = [];
        let output = [];
        for (let i = windowSize; i < array.length - 1; i++) {
            input.push(array.slice(i - windowSize, i));
            output.push(array[i + 1]);
        }

        const xs = tf.tensor2d(input, [input.length, windowSize]);
        const ys = tf.tensor2d(output, [output.length, 1]);

        return [xs, ys];
    }

    // it("Simple prediction Sin", async function (done) {
    //     this.timeout(500000); // This works

    //     // declare models
    //     const model = tf.sequential(); //sequential model is any model where the outputs of one layer are the inputs to the next layer
    //     model.add(tf.layers.lstm({units: 8, inputShape: [1]}));
    //     model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

    //     // Specify loss and optimizer for model
    //     model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

    //     // Prepare training data
    //     const [xs, ys] = generateSinTrainingData(100);

    //     // Train the model
    //     await model.fit(xs, ys, { epochs: 1500 }).then(async () => {
    //         // Use model to predict values
    //         const result = await model.predict(tf.tensor2d([-0.9], [1, 1])).toString();
    //         done();
    //     });
    // })

    // it("Simple prediction Sin with two hidden layers", async function (done) {
    //     this.timeout(500000); // This works

    //     // declare models
    //     const model = tf.sequential(); //sequential model is any model where the outputs of one layer are the inputs to the next layer


    //     // inputShape [1] - one number is in input
    //     model.add(tf.layers.dense({ units: 5, inputShape: [1] })); // hidden layer
    //     model.add(tf.layers.dense({ units: 5, inputShape: [1] })); // hidden layer
    //     model.add(tf.layers.dense({ units: 1})); //output layer

    //     // Specify loss and optimizer for model
    //     model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

    //     // Prepare training data
    //     const [xs, ys] = generateSinTrainingData(100);

    //     // Train the model
    //     await model.fit(xs, ys, { epochs: 1000 }).then(async () => {
    //         // Use model to predict values
    //         console.log(Math.sin(-0.9));
    //         const result = await model.predict(tf.tensor2d([-0.9], [1, 1])).toString();
    //         done();
    //     });
    // })


    it("Simple prediction Sin with LSTM", async function (done) {
        this.timeout(500000); // This works

        // Prepare training data
        let sinTimeSeries = generateSinTimeSeries(100);
        const [xs, ys] = generateTimeSeriesTensor(sinTimeSeries, 5);

        // declare models
        const model = tf.sequential(); 
        model.add(tf.layers.lstm({ units: 10, inputShape: [1]})); // hidden layer
        model.add(tf.layers.dense({ units: 1 })); //output layer

        // Specify loss and optimizer for model
        model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

        // Train the model
        await model.fit(xs, ys, { epochs: 1000 }).then(async () => {
            // Use model to predict values
            console.log(Math.sin(-0.9));
            const result = await model.predict(tf.tensor2d([-0.9], [1, 1])).toString();
            done();
        });
    })
});