import * as rm from "typed-rest-client/RestClient"
const stockdata = require('stock-data.js');

export class StockData {

    private static token = "svdcKVw3fAgEksDD0lxwDheUGjbGSpdFfjOMyBx2HtVwR8AY4z3BCxTTvO1T";

    static async getStockData(symbol: string, date: string, dates: string[] = null): Promise<[number[], string[]]> {
        let returnPromise: Promise<any> = new Promise((resolve, reject) => {
            stockdata.historical({
                symbol: symbol,
                API_TOKEN: this.token,
                options: {
                    date_from: date
                }
            })
                .then((response: { history: any[] }) => {
                    let history = response.history;
                    let apiResult: { value: number, date: string }[] = [];
                    let result = [];

                    if (dates) {
                        for (let key in history) {
                            apiResult.unshift({
                                value: +history[key]["close"],
                                date: key
                            });
                        }

                        for (let dateItem of dates) {
                            let value = apiResult.find(x => x.date === dateItem);
                            if (value) result.push(value.value)
                            else result.push(-1);

                        }
                    } else {
                        dates = [];
                        for (let key in history) {
                            result.unshift(+history[key]["close"]);
                            dates.unshift(key);
                        }
                    }

                    resolve([result, dates]);
                })
                .catch((error: any) => {
                    console.log(error);
                    reject();
                });
        });
        return returnPromise;
    }


    static async getForex(baseSymbol: string, convertTo: string, date: string): Promise<number[]> {
        let returnPromise: Promise<any> = new Promise((resolve, reject) => {
            stockdata.forex.historical({
                base: baseSymbol,
                convert_to: convertTo,
                API_TOKEN: this.token,
                options: {
                    date_from: date
                }
            })
                .then((response: { history: any[] }) => {
                    let history = response.history;
                    let result: number[] = [];

                    for (let key in history) {
                        result.unshift(+history[key]);
                    }

                    resolve(result);
                })
                .catch((error: any) => {
                    console.log(error);
                    reject();
                });
        });
        return returnPromise;

    }

    static getFewAppleMockedData(quantity: number) {
        return this.getAppleMockedData().slice(0, quantity).reverse();
    }


    static getAppleMockedData() {
        return [210.52, 200.67, 204.61, 204.30, 205.28, 207.16, 207.48, 204.53, 203.86,
            203.13, 199.25, 199.23, 198.87, 198.95, 200.62, 199.50, 200.10, 197.00, 195.69, 195.35, 194.02, 191.24,
            189.95, 188.72, 188.47, 186.79, 188.74, 191.05, 195.09, 188.16, 186.53, 188.02, 186.12, 183.73, 181.71,
            180.91, 178.90, 172.91, 172.50, 174.52, 175.53, 175.85, 174.97, 173.15, 174.87, 174.33, 174.23, 172.97,
            171.06, 172.03, 170.93, 170.42, 170.80, 170.18, 170.89, 169.43, 170.41, 170.94, 174.24, 174.18, 171.25,
            166.52, 166.44, 165.25, 154.68, 156.30, 157.76, 152.70, 153.92, 153.30, 156.82, 155.86, 154.94, 153.07,
            150.00, 152.29, 153.80, 153.31, 150.75, 147.93, 148.26, 142.19, 157.92, 157.74, 156.23, 156.15, 157.17,
            146.83, 150.73, 156.83, 160.89, 166.07, 163.94, 165.48, 170.95, 169.10, 168.63, 169.60, 168.49, 174.72,
            176.69, 184.82, 178.58, 179.55, 180.94, 174.24, 174.62, 172.29, 176.78, 176.98, 185.86, 193.53, 191.41,
            186.80, 192.23, 194.17, 204.47, 208.49, 209.95, 203.77, 201.59, 207.48, 222.22, 218.86, 213.30, 212.24,
            216.30, 219.80, 215.09, 222.73, 220.65, 219.31, 216.02, 221.19, 222.15, 217.36, 222.11, 214.45, 216.36,
            226.87, 223.77, 224.29, 227.99, 232.07, 229.28, 227.26, 225.74, 224.95, 220.42, 222.19, 220.79, 217.66,
            220.03, 218.37, 218.24, 217.88, 223.84, 226.41, 221.07, 223.85, 218.33, 221.30, 223.10, 226.87, 228.36,
            227.63, 225.03, 222.98, 219.70, 217.94, 216.16, 215.49, 215.05, 215.04, 215.46, 217.58, 213.32, 210.24,
            209.75, 208.87, 207.53, 208.88, 207.25, 207.11, 209.07, 207.99, 207.39, 201.50, 190.29, 189.91, 190.98,
            194.21, 194.82, 193.00, 191.61, 191.44, 191.88, 190.40, 191.45, 190.91, 191.33, 191.03, 187.88, 190.35,
            190.58, 187.97, 185.40, 183.92, 187.18, 185.11, 185.50, 184.16, 184.43, 182.17, 184.92, 185.46, 186.50,
            185.69, 188.74, 188.84, 190.80, 190.70, 192.28, 191.23, 191.70, 193.46, 193.98, 193.31, 191.83, 190.24,
            186.87, 187.50, 187.90, 188.58, 188.15, 188.36, 187.16, 187.63, 186.31, 186.99, 188.18, 186.44, 188.15,
            188.59, 190.04, 187.36, 186.05, 185.16, 183.83, 176.89, 176.57, 169.10, 165.26, 162.32, 164.22, 163.65,
            162.94, 165.24, 165.72, 172.80, 177.84, 178.24, 175.82, 174.73, 174.14, 172.44, 173.25, 170.05, 168.38,
            172.80, 171.61, 168.39, 166.68, 167.78, 166.48, 168.34, 172.77, 164.94, 168.85, 171.27, 175.24, 175.30,
            178.02, 178.65, 178.44, 179.97, 181.72, 179.98, 176.94, 175.03, 176.67, 176.82, 176.21, 175.00, 178.12,
            178.39, 178.97, 175.50, 172.50, 171.07, 171.85, 172.43, 172.99, 167.37, 164.34, 162.71, 156.41, 155.15,
            159.54, 163.03, 156.49, 160.50, 167.78, 167.43, 166.97, 167.96, 171.51, 171.11, 174.22, 177.04, 177.00,
            178.46, 179.26, 179.10, 176.19, 177.09, 175.28, 174.29, 174.33, 174.35, 175.00, 173.03, 172.23, 172.26,
            169.23, 171.08, 170.60, 170.57, 175.01, 175.01, 174.35, 174.54, 176.42, 173.97, 172.22, 172.27, 171.70,
            172.67, 169.37, 169.32, 169.01, 169.64, 169.80, 171.05, 171.85, 169.48, 173.07, 174.09, 174.97, 174.96,
            173.14, 169.98, 170.15, 171.10, 169.08, 171.34, 173.97, 174.67, 175.88, 176.24, 174.81, 174.25, 172.50,
            168.11, 166.89, 169.04, 166.72, 163.05, 157.41, 156.41, 157.10, 156.17, 156.25, 155.98, 159.76, 160.47,
            159.88, 156.99, 156.00, 156.55, 155.90, 155.84, 155.30, 155.39, 153.48, 154.48, 153.81, 154.12, 153.28,
            154.23, 153.14, 150.55, 151.89, 153.39, 156.07, 158.73, 158.67, 159.88, 158.28, 159.65, 160.86, 161.50,
            158.63, 161.26, 161.91, 162.08, 164.05, 164.00, 163.35, 162.91, 161.47, 159.86, 159.27, 159.98, 159.78,
            157.21, 157.50, 157.86, 160.95, 161.60, 159.85, 157.48, 155.32, 161.06, 160.08, 158.81, 156.39, 155.57,
            157.14, 150.05, 148.73, 149.50, 150.56, 153.46, 152.74, 152.09, 150.27, 150.34, 151.02, 150.08, 149.56,
            149.04, 147.77, 145.74, 145.53, 145.06, 144.18, 142.73, 144.09, 143.50, 144.02, 143.68, 145.83, 143.73,
            145.82, 146.28, 145.63, 145.87, 145.01, 146.34, 142.27, 144.29, 145.16, 146.59, 145.42, 148.98, 154.99,
            155.37, 154.45, 153.93, 155.45, 153.18, 152.76, 153.67, 153.61, 153.87, 153.34, 153.80, 153.99, 153.06,
            152.54, 150.25, 155.47, 155.70, 156.10, 153.95, 153.26, 153.99, 153.01, 148.96, 146.53, 147.06, 147.51,
            146.58, 143.65, 143.79, 143.68, 144.53, 143.64, 142.27, 142.44, 140.68, 141.20, 141.83, 141.05, 141.80,
            141.63, 143.17, 143.34, 143.66, 144.02, 144.77, 143.70, 143.66, 143.93, 144.12, 143.80, 140.88, 140.64,
            140.92, 141.42, 139.84, 141.46, 139.99, 140.69, 140.46, 138.99, 139.20, 139.14, 138.68, 139.00, 139.52,
            139.34, 139.78, 138.96, 139.79, 136.99, 136.93, 136.66, 136.53, 137.11, 136.70, 135.72, 135.35, 135.51,
            135.02, 133.29, 132.12, 132.42, 132.04, 131.53, 130.29, 129.08, 128.53, 128.75, 121.35, 121.63, 121.95,
            121.94, 121.88, 119.97, 120.08, 120.00, 119.78, 119.99, 120.00, 119.04, 119.25, 119.75, 119.11, 118.99,
            117.91, 116.61, 116.02, 116.15, 115.82, 116.73, 116.76, 117.26, 116.52, 116.29, 117.06, 116.95, 116.64,
            115.97, 115.82, 115.19, 115.19, 113.30, 113.95, 112.12, 111.03, 109.95, 109.11, 109.90, 109.49, 110.52,
            111.46, 111.57, 111.79, 111.23, 111.80, 111.73, 110.06, 109.95, 109.99, 107.11, 105.71, 108.43, 107.79,
            110.88, 111.06, 110.41, 108.84, 109.83, 111.59, 111.49, 113.54, 113.72, 114.48, 115.59, 118.25, 117.65,
            116.60, 117.06, 117.12, 117.47, 117.55, 117.63, 116.98, 117.34, 116.30, 116.05, 114.06, 113.89, 113.05,
            113.00, 112.52, 113.05, 112.18, 113.95, 113.09, 112.88, 112.71, 114.62, 113.55, 113.57, 113.58, 114.92,
            115.57, 111.77, 107.95, 105.44, 103.13, 105.52, 108.36, 107.70, 107.73, 106.73, 106.10, 106.00, 106.82,
            106.94, 107.57, 108.03, 108.85, 108.51, 109.36, 109.08, 109.22, 109.38, 109.48, 108.18, 107.93, 108.00,
            108.81, 108.37, 107.48, 105.87, 105.79, 104.48, 106.05, 104.21, 104.34, 102.95, 96.67, 97.34, 98.66,
            99.43, 99.96, 99.87, 99.83, 98.78, 98.79, 96.87, 97.42, 96.98, 96.68, 95.94, 95.53, 94.99, 95.89, 95.60,
            94.40, 93.59, 92.04, 93.40, 96.10, 95.55, 95.91, 95.10, 95.33, 97.55, 97.14, 97.46, 97.34, 98.83, 99.65,
            98.94, 99.03, 98.63, 97.92, 97.72, 98.46, 99.86, 100.35, 100.41, 99.62, 97.90, 96.43, 95.22, 94.20, 94.56,
            93.49, 93.88, 90.52, 90.34, 92.51, 93.42, 92.79, 92.72, 93.24, 94.19, 95.18, 93.64, 93.74, 94.83, 97.82,
            104.35, 105.08, 105.68, 105.97, 107.13, 106.91, 107.48, 109.85, 112.10, 112.04, 110.44, 109.02, 108.66,
            108.54, 110.96, 109.81, 111.12, 109.99, 108.99, 109.56, 107.68, 105.19, 105.67, 106.13, 106.72, 105.91,
            105.92, 105.80, 105.97, 104.58, 102.52, 102.26, 101.17, 101.12, 101.03, 101.87, 103.01, 101.50, 100.75,
            100.53, 96.69, 96.91, 96.76, 96.10, 94.69, 96.88, 96.04, 96.26, 98.12, 96.64, 93.99, 93.70, 94.27, 94.99,
            95.01, 94.02, 96.60, 96.35, 94.48, 96.43, 97.34, 94.09, 93.42, 99.99, 99.44, 101.42, 96.30, 96.79, 96.66,
            97.13, 99.52, 97.39, 99.96, 98.53, 96.96, 96.45, 100.70, 102.71, 105.35, 105.26, 107.32, 108.74, 106.82, 108.03, 108.61, 107.23, 107.33, 106.03, 108.98, 111.34, 110.49, 112.48, 113.18, 116.17, 115.62, 118.23, 118.28, 119.03, 115.20, 116.28, 117.34, 118.30, 117.81, 118.03, 118.88, 117.75, 119.30, 118.78, 117.29, 113.69, 114.18, 112.34, 115.72, 116.11, 116.77, 120.57, 121.06, 120.92, 122.00, 122.57, 121.18, 119.50, 120.53, 119.27, 114.55, 115.28, 119.08, 115.50, 113.76, 113.77, 111.73, 111.04, 111.86, 110.21, 111.79, 111.60, 112.12, 109.50, 110.78, 111.31, 110.78, 110.38, 109.58, 110.30, 109.06, 112.44, 114.71, 115.00, 114.32, 113.40, 115.21, 113.45, 113.92, 116.41, 116.28, 115.31, 114.21, 112.57, 110.15, 112.31, 109.27, 110.37, 112.34, 107.72, 112.76, 113.29, 112.92, 109.69, 103.74, 103.12, 105.76, 112.65, 115.01, 116.50, 117.16, 115.96, 115.15, 115.24, 113.49, 119.72, 115.52, 115.13, 115.40, 114.64, 118.44, 121.30, 122.37, 122.99, 123.38, 122.77, 124.50, 125.16, 125.22, 130.75, 132.07, 129.62, 128.51, 126.82, 125.61, 125.66, 123.28, 120.07, 122.57, 125.69, 126.00, 126.44, 126.60, 125.43, 124.53, 126.75, 127.50, 128.11, 127.03, 127.61, 126.60, 127.88, 127.30, 127.60, 126.92, 127.17, 128.59, 128.88, 127.42, 127.80, 128.65, 129.36, 130.12, 129.96, 130.54, 130.28, 131.78, 132.04, 129.62, 132.54, 131.39, 130.06, 130.07, 130.19, 128.77, 128.95, 126.01, 125.87, 126.32, 127.62, 125.26, 125.01, 125.80, 128.70, 128.95, 125.15, 128.64, 130.56, 132.65, 130.28, 129.67, 128.62, 126.91, 127.60, 124.75, 126.17, 126.78, 126.30, 126.85, 127.10, 126.56, 125.60, 126.01, 127.35, 125.32, 124.25, 124.43, 126.37, 123.25, 124.24, 123.38, 126.69, 127.21, 125.90, 127.50, 128.47, 127.04, 124.95, 123.59, 124.45, 122.24, 124.51, 127.14, 126.60, 126.41, 128.54, 129.36, 129.09, 128.46, 130.42, 128.79, 132.17, 133.00, 129.50, 128.45, 128.72, 127.83, 127.08, 126.46, 124.88, 122.02, 119.72, 118.93, 119.94, 119.56, 118.65, 118.63, 117.16, 118.90, 115.31, 109.14, 113.10, 112.98, 112.40, 109.55, 108.72, 105.99, 106.82, 109.80, 110.22, 109.25, 112.01, 111.89, 107.75, 106.26, 106.25, 109.33, 110.38, 112.52, 113.91, 113.99, 112.01, 112.54, 112.94, 111.78, 112.65, 109.41, 106.75, 108.23, 109.73, 111.62, 111.95, 114.12, 112.40, 115.00, 115.49, 115.93, 114.63, 115.07, 118.93, 119.00, 117.60, 118.63, 116.47, 116.31, 114.67, 115.47, 113.99, 114.18, 112.82, 111.25, 109.70, 108.83, 109.01, 108.70, 108.86, 108.60, 109.40, 108.00, 106.98, 107.34, 106.74, 105.11, 105.22, 104.83, 102.99, 102.47, 99.76, 97.67, 96.26, 97.54, 98.75, 99.81, 100.73, 101.02, 100.80, 98.75, 99.62, 99.62, 99.90, 99.18, 100.75, 100.11, 100.75, 97.87, 101.75, 102.64, 101.06, 100.96, 101.79, 101.58, 100.86, 101.63, 101.66, 101.43, 101.00, 97.99, 98.36, 98.97, 98.12, 98.94, 103.30, 102.50, 102.25, 102.13, 100.89, 101.54, 101.32, 100.58, 100.57, 100.53, 99.16, 97.98, 97.50, 97.24, 95.97, 95.99, 94.74, 94.48, 94.96, 95.12, 95.59, 96.13, 95.60, 98.15, 98.38, 99.02, 97.67, 97.03, 97.19, 94.72, 93.94, 94.43, 93.09, 94.78, 95.32, 96.45, 95.22, 95.04, 95.39, 95.35, 95.97, 94.03, 93.48, 93.52, 92.93, 91.98, 90.90, 90.36, 90.28, 90.83, 90.91, 91.86, 92.18, 92.08, 92.20, 91.28, 92.29, 93.86, 94.25, 93.70, 92.22, 92.48, 92.12, 91.08, 89.81, 90.43, 90.77, 89.14, 89.38, 87.73, 86.75, 86.62, 86.39, 86.37, 85.36, 84.12, 84.84, 84.82, 84.69, 83.65, 84.00, 84.62, 84.92, 85.85, 84.65, 84.50, 84.30, 84.62, 84.87, 81.71, 81.11, 74.96, 75.96, 75.88, 74.99, 74.14, 73.99, 74.53, 74.23, 74.78, 75.76, 74.78, 74.78, 75.97, 76.97, 77.51, 77.38, 76.68, 76.69, 76.78, 77.11, 77.86, 77.03, 76.12, 75.53, 75.89, 75.91, 75.25, 74.96, 75.81, 76.66, 76.58, 75.85, 75.78, 75.82, 76.05, 75.89, 75.39, 75.18, 75.38, 73.91, 74.58, 75.36, 75.04, 75.88, 76.77, 78.00, 77.71, 77.78, 76.56, 76.57, 75.57, 74.24, 73.22, 73.23, 72.68, 71.65, 71.51, 71.40, 71.54, 72.36, 78.64, 78.01, 79.45, 78.79, 78.44, 77.24, 79.18, 79.62, 78.06, 76.53, 76.13, 76.65, 77.64, 77.15, 77.70, 77.28, 79.02, 80.15, 79.22, 80.01, 80.56, 81.10, 81.44, 78.43, 77.78, 78.68, 79.28, 79.64, 79.20, 80.08, 80.19, 80.79, 80.92, 80.00, 81.13, 80.71, 80.90, 78.75, 79.44, 77.99, 76.20, 74.82, 74.26, 74.45, 73.57, 74.22, 74.09, 75.00, 75.45, 74.38, 74.29, 74.15, 74.37, 73.21, 74.42, 75.06, 75.25, 74.29, 74.67, 74.99, 73.81, 75.70, 75.14, 75.99, 74.99, 74.27, 74.48, 72.70, 72.07, 71.59, 71.24, 70.86, 70.40, 69.95, 69.51, 68.71, 69.68, 69.00, 69.06, 69.94, 69.71, 68.11, 68.96, 69.46, 68.79, 69.87, 70.09, 66.77, 67.47, 66.38, 65.05, 64.30, 66.41, 67.53, 66.82, 70.66, 72.31, 71.17, 70.75, 71.24, 69.80, 69.60, 70.24, 70.13, 69.80, 71.85, 71.57, 71.85, 71.77, 71.58, 72.53, 71.76, 71.13, 71.21, 69.94, 66.77, 64.92, 65.86, 66.43, 66.46, 67.06, 66.08, 65.24, 64.65, 64.76, 63.97, 63.00, 62.64, 62.93, 59.86, 60.90, 60.71, 61.68, 61.47, 61.46, 61.06, 60.93, 61.04, 60.10, 60.34, 59.29, 59.63, 60.11, 59.78, 58.46, 56.65, 56.25, 56.87, 57.52, 57.51, 59.07, 59.55, 60.43, 61.68, 61.71, 61.44, 62.28, 61.74, 62.51, 62.70, 63.12, 62.64, 63.59, 64.19, 64.39, 64.25, 64.51, 63.56, 63.06, 63.59, 63.16, 63.05, 62.81, 63.28, 61.89, 62.08, 61.26, 63.41, 64.96, 64.71, 65.25, 66.26, 65.52, 65.82, 64.28, 63.65, 62.76, 63.25, 61.45, 59.60, 58.34, 57.92, 58.02, 56.95, 55.79, 56.01, 57.54, 60.89, 59.98, 61.40, 62.05, 62.24, 61.00, 60.89, 60.46, 61.10, 61.71, 61.40, 61.27, 63.24, 64.58, 65.88, 66.23, 65.99, 64.68, 64.58, 64.93, 65.10, 63.38, 61.79, 61.19, 61.20, 62.55, 61.67, 61.51, 60.81, 61.59, 60.01, 61.50, 63.06, 63.51, 64.14, 63.26, 64.40, 63.72, 64.12, 65.71, 65.74, 66.66, 66.72, 66.84, 68.56, 67.85, 66.89, 65.34, 65.41, 63.19, 64.80, 65.07, 65.26, 65.47, 64.26, 62.84, 64.36, 73.43, 72.11, 71.43, 71.81, 72.30, 69.42, 71.68, 74.33, 74.79, 73.87, 75.04, 74.84, 75.29, 77.44, 78.43, 76.02, 72.80, 73.58, 73.29, 74.31, 74.19, 74.53, 75.19, 76.27, 74.12, 72.83, 75.67, 77.00, 77.34, 75.69, 76.18, 78.18, 76.97, 82.26, 83.74, 83.61, 84.19, 83.28, 83.54, 84.22, 81.64, 80.24, 80.13, 80.82, 75.38, 75.09, 76.70, 77.56, 77.55, 78.15, 76.82, 79.71, 83.26, 83.52, 82.40, 85.22, 85.05, 86.29, 87.08, 88.12, 87.62, 90.58, 87.12, 90.38, 92.09, 92.83, 90.68, 89.96, 89.73, 91.56, 90.84, 91.17, 93.23, 95.26, 95.92, 94.47, 94.20, 95.30, 97.33, 95.03, 96.22, 98.68, 100.01, 99.81, 100.30, 100.27, 99.97, 98.75, 97.57, 95.68, 94.37, 94.68, 97.21, 96.61, 95.75, 96.42, 95.03, 94.84, 96.21, 96.40, 96.53, 94.75, 94.66, 95.55, 93.72, 95.02, 92.59, 90.91, 90.12, 90.24, 90.00, 88.81, 88.68, 88.55, 88.70, 88.94, 87.96, 86.83, 86.69, 87.25, 85.00, 83.59, 82.13, 82.14, 85.85, 86.26, 86.33, 87.76, 86.61, 86.71, 86.70, 86.42, 85.56, 86.35, 86.89, 87.70, 86.55, 87.13, 85.63, 84.65, 83.43, 81.29, 82.07, 81.72, 81.54, 83.16, 82.52, 83.68, 83.92, 83.68, 82.02, 81.65, 81.74, 82.31, 81.60, 82.90, 81.67, 81.64, 80.40, 80.61, 80.14, 82.53, 82.74, 81.75, 80.33, 80.76, 81.51, 79.57, 80.18, 75.77, 75.73, 78.01, 79.02, 79.75, 80.96, 81.50, 81.31, 81.17, 81.35, 80.75, 83.12, 83.71, 83.16, 83.43, 86.14, 86.81, 87.14, 80.04, 81.67, 81.85, 83.92, 86.91, 87.10, 82.88, 86.46, 88.97, 89.46, 89.78, 90.89, 90.53, 89.19, 89.90, 88.38, 85.65, 87.12, 88.23, 87.78, 86.71, 85.15, 85.62, 86.07, 86.57, 85.87, 83.65, 83.65, 84.23, 81.16, 78.86, 77.88, 77.43, 75.81, 75.75, 76.17, 77.88, 77.78, 77.49, 76.49, 75.11, 74.63, 73.77, 73.29, 73.55, 71.73, 71.74, 71.10, 72.78, 71.80, 70.49,
            70.45, 68.10, 66.98, 66.28, 65.67, 65.02, 65.17, 65.21, 64.72, 63.90, 63.52, 63.81, 60.06, 61.06, 60.04,
            61.11, 61.30, 60.67, 59.97, 60.20, 60.36, 60.46, 60.25, 60.34, 59.72, 59.06, 58.75, 57.86, 57.87, 57.52, 58.08, 57.62, 56.94, 56.64, 56.56, 54.60, 54.43, 54.13, 54.31, 55.54, 55.98, 56.23, 55.81, 55.58, 55.85, 56.14, 55.67, 55.42, 54.60, 53.31, 53.73, 51.94, 52.43, 53.79, 52.72, 53.56, 53.92, 54.97, 55.55, 54.18, 54.95, 55.03, 56.47, 58.03, 57.10, 57.18, 57.58, 56.77, 56.64, 57.83, 57.85, 57.81, 57.23, 56.82, 57.97, 56.12, 56.47, 56.95, 60.32, 60.00, 60.29, 58.35, 57.46, 57.18, 55.54, 52.83, 53.91, 54.04, 53.21, 53.51, 54.47, 55.80, 56.72, 57.04, 57.60, 57.76, 57.40, 58.88, 59.06, 58.80, 57.21, 56.14, 55.61, 54.95, 54.28, 53.93, 54.88, 54.85, 54.25, 53.44, 54.43, 54.98, 55.71, 55.71, 54.80, 53.39, 53.74, 53.37, 50.92, 50.86, 52.29, 54.35, 54.35, 54.77, 53.86, 53.39, 51.96, 53.43, 50.46, 53.37, 53.91, 56.08, 55.56, 56.68, 55.78, 55.97, 56.08, 57.63, 56.93, 56.19, 55.33, 55.27, 53.84, 53.40, 52.13, 51.11, 51.15, 50.54, 50.57, 51.39, 51.03, 50.25, 49.92, 49.04, 47.95, 47.72, 47.89, 47.43, 46.62, 47.32, 46.09, 46.47, 45.05, 45.75, 46.45, 46.68, 47.49, 46.66, 46.56, 47.36, 47.46, 47.43, 48.29, 49.06, 49.44, 49.36, 49.69, 48.20, 47.86, 48.11, 47.46, 47.77, 47.89, 48.65, 48.55, 48.02, 47.61, 48.64, 49.51, 49.60, 49.92, 49.66, 49.52, 49.54, 49.94, 49.74, 49.47, 50.02, 49.54, 50.02, 50.06, 50.43, 50.10, 48.92, 48.27, 47.41, 46.78, 47.49, 48.02, 47.49, 47.26, 47.87, 48.30, 48.29, 48.41, 48.74, 49.22, 49.79, 49.80, 50.14, 50.06, 50.22, 49.28, 48.46, 48.74, 48.47, 47.24, 47.81, 47.14, 49.35, 50.51, 50.28, 49.52, 50.35, 50.82, 50.77, 51.43, 51.37, 50.30, 49.90, 50.46, 49.74, 48.98, 48.95, 48.37, 50.08, 51.19, 51.88, 51.41, 51.31, 50.98, 50.65, 51.17, 50.74, 50.27, 49.50, 49.06, 49.19, 49.29, 48.47, 48.01, 49.03, 49.12, 48.77, 48.21, 46.67, 47.53, 48.41, 48.66, 49.78, 49.38, 49.20, 48.81, 48.92, 48.02, 47.68, 47.71, 47.33, 47.08, 46.08, 46.24, 46.47, 46.50, 46.38, 46.23, 46.45, 46.31, 46.03, 45.80, 45.89, 45.77, 45.76, 45.95, 45.79, 45.68, 45.86, 45.46, 45.74, 45.35, 45.45, 45.20, 44.45, 45.27, 45.00, 44.97, 44.10, 44.77, 43.82, 44.06, 42.93, 43.08, 43.86, 44.00, 45.24, 45.43, 45.15, 45.52, 45.30, 45.47, 44.69, 44.19, 43.45, 43.00, 43.61, 43.98, 44.01, 44.12, 43.92, 44.22, 44.36, 44.21, 45.43, 44.96, 43.19, 42.88, 42.65, 42.19, 42.01, 41.32, 41.18, 41.28, 39.81, 40.36, 40.54, 41.05, 40.98, 41.59, 41.76, 41.27, 41.11, 40.54, 40.46, 39.34, 39.51, 38.60, 38.29, 38.15, 37.63, 37.58, 37.56, 36.83, 36.97, 36.02, 35.76, 34.73, 34.64, 34.52, 34.33, 34.70, 34.28, 35.11, 35.66, 35.70, 36.15, 36.00, 35.38, 35.59, 35.97, 35.74, 37.06, 37.39, 37.16, 37.39, 37.57, 37.42, 37.41, 36.75, 36.87, 37.28, 37.73, 37.04, 37.13, 37.00, 36.32, 35.98, 35.08, 35.70, 35.92, 36.10, 35.97, 36.76, 37.09, 36.87, 36.95, 35.52, 35.28, 35.50, 35.93, 36.60, 38.33, 38.10, 38.43, 38.71, 39.12, 38.60, 39.15, 38.84, 38.18, 37.10, 36.33, 36.22, 35.79, 34.74, 35.62, 35.85, 36.57, 37.59, 37.71, 37.26, 36.70, 36.19, 34.87, 35.03, 35.25, 34.62, 33.97, 35.48, 36.05, 36.32, 36.26, 36.91, 37.44, 36.65, 36.28, 33.69, 35.18, 36.57, 36.95, 38.05, 37.30, 38.38, 37.37, 37.43, 38.50, 38.69, 38.07, 37.03, 34.94, 35.30, 35.34, 35.56, 35.10, 34.63, 34.61, 34.54, 34.28, 34.37, 34.22, 34.07, 33.71, 33.57, 33.69, 33.20, 32.99, 32.38, 32.77, 32.62, 32.11, 31.75, 32.09, 32.02, 32.06, 31.98, 32.37, 32.21, 32.12, 31.86, 31.30, 31.28, 30.10, 29.90, 29.84, 29.86, 29.23, 28.86, 28.67, 28.15, 28.63, 28.81, 28.99, 28.94, 29.06, 28.63, 28.38, 27.87, 28.03, 27.73, 27.92, 27.44, 28.46, 27.98, 27.82, 27.44, 28.47, 29.70, 29.42, 29.01, 28.25, 29.72, 30.25, 30.72, 29.42, 29.92, 30.09, 29.67, 30.02, 30.28, 30.08, 30.14, 30.63, 30.57, 30.10, 30.23, 29.87, 30.23, 29.86, 28.87, 28.62, 28.32, 27.92, 27.41, 27.86, 27.74, 28.14, 27.81, 28.06, 28.26, 27.12, 26.99, 27.62, 28.07, 28.03, 28.14, 28.56, 28.66, 29.17, 29.21, 29.41, 28.56, 28.64, 29.42, 29.57, 29.52, 29.21, 28.86, 29.04, 29.00, 28.78, 27.76, 27.72, 27.26, 26.96, 27.04, 26.93, 28.05, 27.49, 28.20, 28.93, 29.13, 29.31, 29.27, 28.39, 27.12, 26.86, 27.22, 27.33, 27.15, 27.26, 27.21, 27.04, 27.18, 27.14, 26.57, 26.41, 25.84, 26.48, 26.48, 26.59, 26.05, 26.26, 26.50, 26.35, 26.29, 26.43, 26.36, 25.98, 25.02, 24.82, 24.59, 24.65, 24.45, 24.70, 24.33, 23.79, 23.60, 23.61, 24.03, 24.29, 24.21, 23.92, 24.20, 24.15, 24.17, 23.76, 23.51, 23.43, 22.80, 23.83, 24.06, 23.62, 23.26, 23.53, 23.64, 23.42, 23.59, 23.65, 23.78, 23.34, 23.26, 22.86, 22.86, 22.87, 22.86, 22.55, 22.39, 21.64, 21.84, 21.68, 21.07, 20.98, 20.32, 20.33, 19.79, 19.48, 19.60, 19.34, 19.80, 20.00, 20.40, 20.35, 20.28, 20.35, 19.98, 19.46, 19.14, 19.62, 19.93, 19.41, 19.37, 19.48, 19.44, 19.57, 19.99, 20.04, 20.39, 20.55, 20.67, 20.53, 20.14, 19.93, 19.91, 19.40, 19.30, 19.01, 18.68, 17.50, 17.74, 17.98, 18.21, 18.09, 17.49, 17.56, 17.07, 17.77, 18.51, 18.46, 18.44, 18.93, 18.96, 18.87, 18.18, 17.98, 17.88, 17.70, 17.82, 17.70, 17.91, 17.36, 17.39, 17.21, 17.63, 17.35, 16.81, 16.90, 17.17, 17.08, 16.62, 16.43, 16.92, 16.57, 16.10, 15.53, 15.02, 14.93, 15.26, 15.70, 15.21, 15.21, 15.38, 14.51, 14.52, 14.50, 14.24, 13.63, 13.70, 13.76, 13.24, 12.66, 11.87, 12.19, 12.69, 13.02, 12.62, 12.56, 12.76, 12.74, 13.02, 12.89, 12.42, 13.03, 12.95, 13.48, 13.50, 14.17, 14.18, 13.83, 13.98, 14.64, 14.25, 13.78, 13.36, 13.28, 13.07, 12.88, 13.29, 13.46, 12.96, 12.81, 12.62, 12.62, 11.83, 11.17, 11.76, 11.91, 12.19, 12.53, 12.67, 12.94, 13.24, 13.00, 13.29, 13.51, 12.96, 12.19, 12.33, 12.37, 12.26, 12.15, 12.34, 12.25, 12.86, 12.78, 12.74, 13.63, 13.54, 14.04, 13.57, 14.03, 14.29, 14.25, 13.43, 13.06, 13.70, 13.21, 12.70, 13.24, 13.57, 12.97, 13.28, 11.80, 11.50, 12.33, 12.84, 12.59, 12.89, 13.78, 12.87, 13.54, 13.70, 14.03, 14.16, 14.76, 15.86, 15.28, 15.37, 15.86, 14.94, 14.27, 13.16, 13.77, 14.03, 13.84, 13.07, 14.06, 13.91, 14.56, 13.99, 14.87, 15.75, 13.83, 12.68, 12.83, 12.74, 14.02, 13.87, 14.30, 15.59, 16.24, 15.04, 18.32, 18.85, 18.39, 18.12, 18.72, 20.13, 19.16, 18.26, 19.98, 20.05, 21.28, 21.81, 21.66, 21.67, 22.56, 22.88, 23.03, 23.85, 23.74, 24.22, 24.82, 24.95, 24.81, 24.65, 25.26, 24.90, 25.12, 24.79, 25.06, 25.11, 25.62, 25.61, 25.25, 24.79, 24.22, 23.37, 23.46, 22.95, 21.89, 22.38, 22.71, 22.84, 22.44, 22.06, 23.16, 22.72, 23.75, 23.15, 23.76, 23.59, 24.54, 24.69, 24.23, 24.84, 24.65, 25.23, 24.89, 25.65, 25.02, 24.30, 24.03, 24.95, 23.92, 24.30, 24.04, 25.34, 24.75, 24.74, 25.04, 25.84, 25.54, 25.92, 25.26, 24.62, 24.75, 25.83, 26.52, 25.94, 26.52, 27.06, 26.46, 26.48, 26.59, 26.96, 26.67, 26.72, 26.63, 25.88, 25.29, 25.46, 26.56, 26.23, 26.80, 27.10, 26.61, 27.14, 26.88, 26.21, 26.44, 26.08, 26.67, 26.39, 25.85, 25.71, 24.85, 25.01, 24.61, 24.25, 24.13, 23.27, 22.89, 24.02, 23.01, 22.07, 21.96, 21.20, 21.11, 21.02, 22.08, 21.63, 21.83, 22.27, 21.87, 21.66, 21.07, 21.36, 20.50, 20.43, 20.04, 20.72, 20.14, 19.93, 19.04, 18.52, 18.97, 18.10, 18.09, 18.28, 18.00, 18.19, 17.10, 17.46, 17.28, 17.78, 17.80, 17.39, 17.86, 18.56, 17.57, 17.02, 17.11, 17.07, 17.36, 17.69, 17.45, 17.80, 18.21, 18.49, 17.84, 18.49, 17.93, 17.32, 17.43, 18.48, 18.81, 19.11, 19.34, 18.88, 18.79, 18.57, 18.57, 19.37, 19.87, 22.23, 23.05, 22.98, 22.81, 24.15, 25.54, 24.67, 25.43, 25.63, 24.46, 25.38, 25.72, 27.85, 27.83, 28.30, 28.55, 28.37, 28.42, 28.40, 27.70, 26.74, 26.16, 26.14, 26.34, 27.20, 27.40, 27.27, 26.93, 27.74, 27.76, 27.14, 26.50, 25.69, 25.55, 26.03, 26.33, 25.75, 24.97, 24.65, 24.51, 24.07, 24.12, 23.42, 23.77, 23.47, 23.73, 24.28, 21.97, 23.62, 25.07, 26.61, 27.40, 26.60, 26.84, 26.78, 27.14, 26.71, 26.44, 26.39, 26.11, 26.56, 26.59, 24.91, 24.35, 24.79, 24.68, 24.23, 23.85, 23.89, 23.18, 23.83, 23.98, 23.99, 23.06, 22.32, 22.56, 22.64, 22.33, 21.92, 22.07, 21.82, 21.88, 21.18, 20.59, 20.04, 20.11, 20.13, 19.77, 19.83, 19.60, 19.55, 19.36, 19.53, 18.82, 19.29, 19.54, 20.59, 19.78, 19.46, 19.15, 18.12, 18.89, 19.33, 18.72, 18.93, 18.22, 17.46, 17.44, 16.72, 17.13, 17.72, 18.26, 17.86, 18.06, 19.14, 19.29, 19.32, 18.84, 19.50, 19.29, 18.82, 20.20, 20.55, 20.86, 19.61, 19.27, 20.53, 20.54, 20.00, 19.73, 19.84, 19.73, 19.68, 19.15, 18.91, 18.91, 18.62, 18.90, 18.96, 18.17, 17.32, 17.43, 17.22, 17.41, 17.09, 17.48, 17.57, 17.70, 17.36, 17.67, 17.87, 17.21, 16.96, 16.79, 17.20, 17.17, 17.78, 17.72, 17.66, 17.52, 17.33, 16.91, 17.31, 16.97, 16.34, 16.23, 15.81, 16.13, 16.22, 16.00, 15.72, 15.63, 15.33, 15.36, 15.62, 15.53, 15.33, 15.27, 15.01, 14.85, 14.40, 14.34, 14.34, 14.21, 14.26, 14.27, 14.12, 13.62, 13.32, 13.36, 13.00, 12.90, 12.91, 12.91, 13.06, 12.89, 13.17, 13.23, 13.46, 13.38, 13.53, 13.47, 13.50, 13.38, 13.27, 13.39, 13.32, 13.64, 13.69, 13.36, 13.42, 13.41, 13.07, 13.02, 12.80, 12.80, 12.86, 12.63, 12.84, 12.57, 12.57, 12.53, 12.60, 12.33, 12.20, 12.44, 12.09, 11.99, 12.66, 12.72, 12.79, 12.74, 12.27, 12.12, 12.17, 12.19, 12.09, 12.13, 11.90, 12.31, 12.31, 12.02, 11.99, 12.11, 12.11, 12.25, 12.22, 12.28, 12.20, 12.32, 12.39, 12.24, 12.40, 12.64, 12.72, 13.56, 13.87, 13.52, 13.69, 13.86, 13.22, 12.21, 12.15, 12.24, 11.97, 12.12, 11.55, 11.65, 11.64, 11.74, 11.84, 12.11, 12.33, 12.21, 12.53, 12.65, 12.72, 12.31, 12.68, 12.61, 12.43, 12.83, 13.04, 13.02, 13.05, 13.09, 13.11, 13.12, 12.79, 13.09, 12.90, 12.66, 12.35, 12.26, 12.23, 12.11, 12.14, 12.05, 11.87, 11.91, 11.78, 11.50, 11.39, 11.18, 11.28, 11.31, 11.58, 11.49, 11.49, 11.74, 11.67, 11.58, 11.64, 11.42, 11.28, 10.65, 10.61, 10.77, 10.72, 10.75, 10.46, 10.54, 10.66, 10.60, 10.69, 10.77, 10.58, 10.69, 11.00, 11.00, 10.92, 11.09, 10.82, 10.43, 10.66, 10.75, 10.54, 10.56, 10.59, 10.60, 10.60, 10.38, 10.36, 10.36, 10.40, 10.00, 10.21, 9.77, 9.69, 9.57, 9.50, 9.57, 9.82, 9.69, 9.62, 9.66, 9.51, 9.70, 9.66, 9.71, 9.49, 9.13, 9.09, 9.15, 9.08, 9.25, 9.60, 9.76, 9.94, 9.74, 9.60, 9.71, 9.37, 9.06, 9.12, 8.85, 8.77, 8.67, 8.64, 7.73, 7.56, 7.48, 7.24, 7.46, 7.57, 7.95, 7.86, 7.91, 7.97, 8.14, 8.28, 8.18, 8.42, 8.00, 8.20, 8.43, 8.40, 8.51, 8.27, 8.21, 8.17, 8.22, 8.48, 8.23, 8.33, 8.14, 8.46, 8.68, 8.37, 8.53, 8.57, 8.81, 8.88,
            8.54, 8.75, 9.08, 9.19, 9.05, 9.02, 9.05, 9.22, 9.03, 9.32, 9.28, 9.68, 9.67, 9.74, 10.09, 10.15, 10.27, 10.27, 10.16, 10.16, 10.23, 9.94, 10.06, 9.91, 9.74, 9.45, 9.39, 9.58, 9.66, 9.38, 9.46, 9.26, 9.50, 9.53, 9.71, 9.81, 9.97, 10.18, 9.60, 8.74, 8.95, 8.96, 8.96, 8.90, 8.39, 8.50, 8.57, 8.59, 8.81, 8.83, 9.14, 9.24, 9.19, 9.46, 9.62, 9.38, 9.03, 9.13, 9.38, 9.47, 9.35, 9.67, 9.94, 9.87, 9.78, 10.14, 10.21, 10.25, 10.19, 9.87, 10.04, 10.08, 9.89, 9.66, 9.24, 9.62, 9.28, 9.83, 9.66, 9.61, 10.26, 10.30, 10.77, 10.79, 10.71, 10.29, 10.33, 10.60, 10.86, 11.10, 10.87, 11.29, 11.78, 12.10, 12.23, 12.04, 11.99, 11.55, 10.86, 10.90, 10.63, 10.71, 10.68, 10.27, 10.21, 10.51, 10.60, 10.48, 10.57, 10.50, 10.30, 10.20, 10.16, 10.31, 10.29, 10.71, 10.70, 10.62, 10.58, 10.56, 10.58, 10.26, 10.38, 10.23, 9.69, 9.73, 9.95, 9.91, 9.59, 9.50, 9.28, 9.22, 9.22, 9.28, 8.90, 8.78, 8.79, 8.74];
    }
}