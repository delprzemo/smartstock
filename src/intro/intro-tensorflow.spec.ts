import { expect } from 'chai';
import * as tf from '@tensorflow/tfjs-node'
import { Tensor } from '@tensorflow/tfjs-node';
import { StockData } from '../data/stock-data.service'
import { CheckModel } from '../data/models/check.model';
import { CheckStatsModel } from '../data/models/check-stats.model';
const jsonfile = require('jsonfile')


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


    // it("Simple prediction Sin with LSTM", async function (done) {
    //     this.timeout(50000000); // This works

    //     const windowSize = 5;
    //     const epochs = 30;
    //     const learningRate = 0.001;
    //     const layers = 2;
    //     const checkIteration = 40;


    //     // Prepare training data
    //     let sinTimeSeries = generateSinTimeSeries(140);
    //     let [normalizedData, min, max] = normalize(sinTimeSeries);
    //     let [input, output] = generateTimeSeriesInputOutput(normalizedData, windowSize);
    //     let [checkModels, trainInput, trainOutput] = splitTrainIteration(input, output, checkIteration);


    //     const trainingResult = await trainModel(trainInput, trainOutput, windowSize, epochs,
    //         learningRate, layers, () => { });


    //     for (let item of checkModels) {
    //         let predicted = await Predict(item, trainingResult.model)[0];
    //         let deNormalized = deNormalize(predicted, min, max);
    //         item = calculateCheckValues(item, deNormalized);
    //     }

    //     console.log(calculateStatistics(checkModels));
    // })


    // it("Simple prediction Stock Data with LSTM", async function (done) {
    //     this.timeout(50000000); // This works

    //     const windowSize = 10;
    //     const epochs = 20;
    //     const learningRate = 0.001;
    //     const layers = 2;
    //     const checkIteration = 10;


    //     // Prepare training data
    //     let stockTimeSeries = StockData.getFewAppleMockedData(110);
    //     let [normalizedData, min, max] = normalize(stockTimeSeries);
    //     let [input, output] = generateTimeSeriesInputOutput(normalizedData, windowSize);
    //     let [checkModels, trainInput, trainOutput] = splitTrainIteration(input, output, checkIteration);


    //     const trainingResult = await trainModel(trainInput, trainOutput, windowSize, epochs,
    //         learningRate, layers, () => { });


    //     for (let item of checkModels) {
    //         let predicted = await Predict(item, trainingResult.model)[0];
    //         let deNormalized = deNormalize(predicted, min, max);
    //         item = calculateCheckValues(item, deNormalized, min, max);
    //     }

    //     console.log(calculateStatistics(checkModels));

    // })



    // it("Simple prediction Real Stock Data with LSTM", async function (done) {
    //     this.timeout(50000000); // This works

    //     const windowSize = 5;
    //     const epochs = 25;
    //     const learningRate = 0.001;
    //     const layers = 2;
    //     const checkIteration = 10;


    //     // Prepare training data
    //     let stockTimeSeries = await StockData.getStockData("AAPL", "2019-01-01");
    //     let [normalizedData, min, max] = normalize(stockTimeSeries);
    //     let [input, output] = generateTimeSeriesInputOutput(normalizedData, windowSize);
    //     let [checkModels, trainInput, trainOutput] = splitTrainIteration(input, output, checkIteration);


    //     const trainingResult = await trainModel(trainInput, trainOutput, windowSize, epochs,
    //         learningRate, layers, () => { });


    //     for (let item of checkModels) {
    //         let predicted = await Predict(item, trainingResult.model)[0];
    //         let deNormalized = deNormalize(predicted, min, max);
    //         item = calculateCheckValues(item, deNormalized, min, max);
    //     }

    //     console.log(calculateStatistics(checkModels));

    // })


    it("Simple prediction for multiple Real Stock Data with LSTM", async function (done) {
        this.timeout(50000000);

        let date = '2018-01-01';
        let allSymbols = StockData.getAllStockSymbols();
        let result = 0;

        // saveToFile({
        //     currentDate: new Date().toLocaleDateString(),
        //     date: date,
        //     multiple: false
        // });

        // for(let symbol of allSymbols) {
        //     result = result + (await trainAndCheck(symbol, date)).avgError;
        // }

        // saveToFile({
        //     errSum: result / allSymbols.length
        // });

        saveToFile({
            currentDate: new Date().toLocaleDateString(),
            date: date,
            multiple: true
        });

        result = 0;

        for (let symbol of allSymbols) {
            result = result + (await trainAndCheck(symbol, date, ...allSymbols)).avgError;
        }

        saveToFile({
            errSum: result / allSymbols.length
        });

    })


    // it("Just verify Real Stock Data with LSTM", async function (done) {
    //     this.timeout(50000000);

    //     let date = '2018-10-01';
    //     let allSymbols = StockData.getAllStockSymbols();
    //     let result = 0;

    //     result = result + (await trainAndCheck("AAPL", date)).avgError;


    // })


    // it("Simple prediction for multiple Forex Stock Data with LSTM", async function (done) {
    //     this.timeout(50000000);

    //     let date = '2018-01-01';
    //     let allSymbols = StockData.getAllForexSymbols();
    //     let result = 0;

    //     saveToFile({
    //         currentDate: new Date().toLocaleDateString(),
    //         date: date,
    //         multiple: false
    //     });

    //     for(let symbol of allSymbols) {
    //         result = result + (await trainAndCheckForex([symbol, "USD"], date)).avgError;
    //     }

    //     saveToFile({
    //         errSum: result / allSymbols.length
    //     });

    //     saveToFile({
    //         currentDate: new Date().toLocaleDateString(),
    //         date: date,
    //         multiple: true
    //     });

    //     result = 0;
    //     let restSymbols: [string, string][] = [];

    //     for(let symbol of allSymbols) {
    //         restSymbols.push([symbol, "USD"])
    //     }

    //     for(let symbol of allSymbols) {
    //         result = result + (await trainAndCheckForex([symbol, "USD"], date, ...restSymbols)).avgError;
    //     }

    //     saveToFile({
    //         errSum: result / allSymbols.length
    //     });
    // })



    async function trainAndCheckForex(baseSymbol: [string, string], date: string,
        ...rest: [string, string][]): Promise<CheckStatsModel> {

        let returnPromise: Promise<CheckStatsModel> = new Promise(async (resolve, reject) => {
            // configuration
            const windowSize = 10;
            const epochs = 100 * (rest.length + 1);
            const learningRate = 0.001;
            const layers = 2;
            const checkIteration = 10;
            let restStocks = [];
            let restStockAfterFilter = [];
            const move: number = 7;

            // Prepare training data
            let [leftSymbol, rightSymbol] = baseSymbol;
            let [stockTimeSeries, dates] = await StockData.getForex(leftSymbol, rightSymbol, date);

            // get rest stock data
            for (let item of rest) {
                let [leftRestSymbol, rightRestSymbol] = item;
                let [stockData, stockDates] = await StockData.getForex(leftRestSymbol, rightRestSymbol, date, dates);
                restStocks.push(stockData);
            }

            // remove missing data
            let missingDataIndexes = getMissingIndexes(restStocks);

            for (let item of restStocks) {
                item = removeEmptyData(item, missingDataIndexes);
                restStockAfterFilter.push(justNormalize(item));
            }

            stockTimeSeries = removeEmptyData(stockTimeSeries, missingDataIndexes);

            // preapare input and output
            let [normalizedData, min, max] = normalize(stockTimeSeries);
            let [input, output] = generateTimeSeriesInputOutput(normalizedData, windowSize, move, ...restStockAfterFilter);
            let [predictionInput, predictionOutput] = generatePredictionInputOutput(normalizedData, windowSize, ...restStockAfterFilter);
            let [checkModels, trainInput, trainOutput] = splitTrainIteration(input, output, checkIteration);
            let sumWindowSize = windowSize * (1 + restStockAfterFilter.length);

            // train
            const trainingResult = await trainModel(trainInput, trainOutput, sumWindowSize, epochs,
                learningRate, layers, () => { });


            // predict
            for (let item of checkModels.filter(x => x.output !== null)) {
                let predicted = await Predict(item, trainingResult.model)[0];
                let deNormalized = deNormalize(predicted, min, max);
                item = calculateCheckValues(item, deNormalized, min, max, restStockAfterFilter.length);
            }

            // prediction
            let prediction = await Predict(new CheckModel(predictionInput[0]), trainingResult.model)[0];
            let deNormalizedPrediction = deNormalize(prediction, min, max);

            // check
            let stats = calculateStatistics(checkModels.filter(x => x.output !== null));
            let lastValue = checkModels[checkModels.length - 1].input[checkModels[0].input.length - 1];
            let denormalizedLastValue = deNormalize(lastValue, min, max);
            saveToFile({
                mainStock: baseSymbol,
                stats: stats,
                invest: deNormalizedPrediction > denormalizedLastValue,
                restStocks: rest,
                date: date,
                options: {
                    windowSize: windowSize,
                    learningRate: learningRate,
                    epochs: epochs,
                    layers: layers
                },
                forex: true
            })

            resolve(stats);

        });

        return returnPromise;

    }

    async function trainAndCheck(baseSymbol: string, date: string, ...rest: string[]): Promise<CheckStatsModel> {

        let returnPromise: Promise<CheckStatsModel> = new Promise(async (resolve, reject) => {
            // configuration
            const windowSize = 51;
            const epochs = 120 * (rest.length + 1);
            const learningRate = 0.001;
            const layers = 2;
            const checkIteration = 10;
            let restStocks = [];
            let restStockAfterFilter = [];
            const move: number = 7;

            // prepare training data
            let [stockTimeSeries, dates] = await StockData.getStockData(baseSymbol, date);

            // get rest stock data
            for (let item of rest) {
                let [stockData, stockDates] = await StockData.getStockData(item, date, dates);
                restStocks.push(stockData);
            }

            // remove missing data
            let missingDataIndexes = getMissingIndexes(restStocks);

            for (let item of restStocks) {
                item = removeEmptyData(item, missingDataIndexes);
                restStockAfterFilter.push(justNormalize(item));
            }

            stockTimeSeries = removeEmptyData(stockTimeSeries, missingDataIndexes);

            // preapare input and output
            let [normalizedData, min, max] = normalize(stockTimeSeries);
            let [input, output] = generateTimeSeriesInputOutput(normalizedData, windowSize, move, ...restStockAfterFilter);
            let [predictionInput, predictionOutput] = generatePredictionInputOutput(normalizedData, windowSize, ...restStockAfterFilter);
            let [checkModels, trainInput, trainOutput] = splitTrainIteration(input, output, checkIteration);
            let sumWindowSize = windowSize * (1 + restStockAfterFilter.length);

            // train
            const trainingResult = await trainModel(trainInput, trainOutput, sumWindowSize, epochs,
                learningRate, layers, () => { });


            // predict check
            for (let item of checkModels.filter(x => x.output !== null)) {
                let predicted = await Predict(item, trainingResult.model)[0];
                let deNormalized = deNormalize(predicted, min, max);
                item = calculateCheckValues(item, deNormalized, min, max, restStockAfterFilter.length);
            }

            // prediction
            let prediction = await Predict(new CheckModel(predictionInput[0]), trainingResult.model)[0];
            let deNormalizedPrediction = deNormalize(prediction, min, max);

            // check
            let stats = calculateStatistics(checkModels.filter(x => x.output !== null));
            let lastValue = checkModels[checkModels.length - 1].input[checkModels[0].input.length - 1];
            saveToFile({
                mainStock: baseSymbol,
                stats: stats,
                invest: deNormalizedPrediction > lastValue,
                restStocks: rest,
                date: date,
                options: {
                    windowSize: windowSize,
                    learningRate: learningRate,
                    epochs: epochs,
                    layers: layers
                },
                forex: false
            })

            resolve(stats);

        });

        return returnPromise;

    }

    function removeEmptyData(data: number[], indexes: number[]) {
        let result = [];
        for (let key in data) {
            if (!indexes.some(x => x === +key)) {
                result.push(data[key]);
            }
        }

        return result;
    }

    function getMissingIndexes(data: number[][]): number[] {
        let result = [];

        for (let dataItem of data) {
            for (let key in dataItem) {
                if (dataItem[key] < 0) result.push(+key);
            }
        }
        return result;
    }

    function normalize(input: number[]): [number[], number, number] {

        const min = Math.min.apply(null, input);
        const max = Math.max.apply(null, input);

        input = input.map(x => x = (x - min) / (max - min));

        return [input, min, max];
    }

    function justNormalize(input: number[]): number[] {

        const min = Math.min.apply(null, input);
        const max = Math.max.apply(null, input);

        input = input.map(x => x = (x - min) / (max - min));

        return input;
    }

    function deNormalize(value: number, min: number, max: number) {
        return (value * (max - min)) + min;
    }


    async function trainModel(inputs: any[], outputs: any[], window_size: number, n_epochs: number,
        learning_rate: number, n_layers: number, callback: Function) {
        const input_layer_shape = window_size;
        const input_layer_neurons = 20;

        const rnn_input_layer_features = 10;
        const rnn_input_layer_timesteps = input_layer_neurons / rnn_input_layer_features;

        const rnn_input_shape = [rnn_input_layer_features, rnn_input_layer_timesteps];
        const rnn_output_neurons = 20;

        const rnn_batch_size = window_size;

        const output_layer_shape = rnn_output_neurons;
        const output_layer_neurons = 1;

        const model = tf.sequential();

        const xs = tf.tensor2d(inputs, [inputs.length, inputs[0].length]).div(tf.scalar(10));
        const ys = tf.tensor2d(outputs, [outputs.length, 1]).reshape([outputs.length, 1]).div(tf.scalar(10));

        model.add(tf.layers.dense({ units: input_layer_neurons, inputShape: [input_layer_shape] }));
        model.add(tf.layers.reshape({ targetShape: rnn_input_shape }));

        var lstm_cells = [];
        for (let index = 0; index < n_layers; index++) {
            lstm_cells.push(tf.layers.lstmCell({ units: rnn_output_neurons }));
        }

        model.add(tf.layers.rnn({
            cell: lstm_cells,
            inputShape: rnn_input_shape, returnSequences: false
        }));

        model.add(tf.layers.dense({ units: output_layer_neurons, inputShape: [output_layer_shape] }));

        const opt_adam = tf.train.adam(learning_rate);
        model.compile({ optimizer: opt_adam, loss: 'meanSquaredError' });

        const hist = await model.fit(xs, ys,
            {
                batchSize: rnn_batch_size, epochs: n_epochs, callbacks: {
                    onEpochEnd: async (epoch, log) => { callback(epoch, log); }
                }
            });

        return { model: model, stats: hist };
    }

    function calculateStatistics(models: CheckModel[]): CheckStatsModel {
        let budget = 1000;
        let sumError = 0;
        let correctedTrends = 0;
        for (let item of models) {
            sumError = sumError + item.error;
            correctedTrends = correctedTrends + (item.isCorrectTrend ? 1 : 0);
            if (item.calculatedOutput > item.input[item.input.length - 1]) {
                budget = budget * (item.output / item.input[item.input.length - 1])
            }
        }

        const obj = new CheckStatsModel(sumError / models.length,
            correctedTrends / models.length,
            budget)

        return obj;
    }

    function saveToFile(obj: Object) {

        const file = '/tmp/summary.json'
        jsonfile.writeFile(file, obj, { flag: 'a' }, function (err: any) {
            if (err) console.error(err)
        })
    }

    function calculateCheckValues(model: CheckModel, predictedValue: number,
        min: number, max: number, numExtraStocks: number): CheckModel {

        model.output = deNormalize(model.output, min, max);
        model.input = model.input
            .slice(0, model.input.length / (numExtraStocks + 1))
            .map(x => deNormalize(x, min, max));

        model.calculatedOutput = predictedValue;
        model.error = Math.abs(model.output - model.calculatedOutput);
        let isCorrectTrend = true;
        let lastInputValue = model.input[model.input.length - 1];

        if (lastInputValue < model.output) {
            isCorrectTrend = lastInputValue < model.calculatedOutput
        }
        if (lastInputValue > model.output) {
            isCorrectTrend = lastInputValue > model.calculatedOutput
        }

        model.isCorrectTrend = isCorrectTrend;
        return model;
    }

    function splitTrainIteration(input: number[][], output: number[],
        checkIteration: number): [CheckModel[], number[][], number[]] {

        let trainInput = input.slice(0, input.length - checkIteration - 1);
        let trainOutput = output.slice(0, input.length - checkIteration - 1);
        let result: CheckModel[] = [];

        for (let i = input.length - checkIteration - 1; i < input.length - 1; i++) {
            result.push(new CheckModel(input[i], output[i]));
        }

        return [result, trainInput, trainOutput];
    }

    function Predict(checkModel: CheckModel, model: tf.Sequential) {

        let inps = [checkModel.input];

        const outps = (model.predict(tf.tensor2d(inps, [inps.length,
        inps[0].length]).div(tf.scalar(10))) as Tensor).mul(10);

        return Array.from(outps.dataSync());
    }


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

    function generatePredictionInputOutput(array: number[],
        windowSize: number, ...rest: number[][]): [number[][], number[]] {

        let input = [];
        let output = [];
        let i = array.length;
        let toPush = array.slice(i - windowSize, i);

        for (let serie of rest) {
            toPush = toPush.concat(serie.slice(i - windowSize, i));
        }

        input.push(toPush);
        output.push(null);

        return [input, output];
    }

    function generateTimeSeriesInputOutput(array: number[],
        windowSize: number, move: number, ...rest: number[][]): [number[][], number[]] {

        let input = [];
        let output = [];
        for (let i = windowSize; i <= array.length - move; i++) {
            let toPush = array.slice(i - windowSize, i);

            for (let serie of rest) {
                toPush = toPush.concat(serie.slice(i - windowSize, i));
            }

            input.push(toPush);
            output.push(array[i + move]);
        }

        return [input, output];
    }
});