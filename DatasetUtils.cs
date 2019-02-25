using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace PfiTest
{
    public static class DatasetUtils
    {
        private static string Download(string baseGitPath, string dataFile)
        {
            using (WebClient client = new WebClient())
            {
                client.DownloadFile(new Uri($"{baseGitPath}"), dataFile);
            }

            return dataFile;
        }
        
        /// <summary>
        /// Downloads the housing dataset from the ML.NET repo.
        /// </summary>
        public static string DownloadHousingRegressionDataset()
        {
            var fileName = "housing.txt";
            if (!File.Exists(fileName))
                Download("https://raw.githubusercontent.com/dotnet/machinelearning/024bd4452e1d3660214c757237a19d6123f951ca/test/data/housing.txt", fileName);
            return fileName;
        }

        public static IDataView LoadHousingRegressionDataset(MLContext mlContext)
        {
            // Download the file
            string dataFile = DownloadHousingRegressionDataset();

            // Define the columns to read
            var reader = mlContext.Data.CreateTextLoader(
                columns: new[]
                    {
                        new TextLoader.Column("MedianHomeValue", DataKind.R4, 0),
                        new TextLoader.Column("CrimesPerCapita", DataKind.R4, 1),
                        new TextLoader.Column("PercentResidental", DataKind.R4, 2),
                        new TextLoader.Column("PercentNonRetail", DataKind.R4, 3),
                        new TextLoader.Column("CharlesRiver", DataKind.R4, 4),
                        new TextLoader.Column("NitricOxides", DataKind.R4, 5),
                        new TextLoader.Column("RoomsPerDwelling", DataKind.R4, 6),
                        new TextLoader.Column("PercentPre40s", DataKind.R4, 7),
                        new TextLoader.Column("EmploymentDistance", DataKind.R4, 8),
                        new TextLoader.Column("HighwayDistance", DataKind.R4, 9),
                        new TextLoader.Column("TaxRate", DataKind.R4, 10),
                        new TextLoader.Column("TeacherRatio", DataKind.R4, 11),
                    },
                hasHeader: true
            );

            // Read the data into an IDataView
            var data = reader.Read(dataFile);

            return data;
        }

        /// <summary>
        /// A class to hold the raw housing regression rows.
        /// </summary>
        public sealed class HousingRegression
        {
            public float MedianHomeValue { get; set; }
            public float CrimesPerCapita { get; set; }
            public float PercentResidental { get; set; }
            public float PercentNonRetail { get; set; }
            public float CharlesRiver { get; set; }
            public float NitricOxides { get; set; }
            public float RoomsPerDwelling { get; set; }
            public float PercentPre40s { get; set; }
            public float EmploymentDistance { get; set; }
            public float HighwayDistance { get; set; }
            public float TaxRate { get; set; }
            public float TeacherRatio { get; set; }
        }
    }
}
