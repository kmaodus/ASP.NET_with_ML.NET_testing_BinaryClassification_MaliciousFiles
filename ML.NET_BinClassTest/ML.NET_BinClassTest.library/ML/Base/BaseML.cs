using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.NET_BinClassTest.library.ML.Base
{
    public class BaseML
    {
        protected const string FEATURES = "Features";

        protected MLContext MlContext;

        public BaseML()
        {
            MlContext = new MLContext(2021);
        }
    }
}
