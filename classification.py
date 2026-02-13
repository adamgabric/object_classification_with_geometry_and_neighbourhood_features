import os
import joblib
import numpy as np
import sklearn
import sklearn.preprocessing
import sklearn.neighbors
import sklearn.tree
import sklearn.gaussian_process
import sklearn.svm
import sklearn.ensemble
import sklearn.neural_network
import sklearn.metrics
import imblearn.over_sampling
import imblearn.under_sampling
import imblearn.pipeline
import shapely
import pandas as pd
import geopandas as gpd

from rasterio import features
from rasterio.transform import Affine
from pathlib import Path


class ObjectClassification:
    def __init__(self, gdf, columns, features_type='geom_features', model='rf', model_parameters=None, sampling=False,
                 training=False, model_folder='', ground_truth=None, test_areas=None, validation_areas=None,
                 fixed_test_area=True, run=None):
        self.gdf = gdf
        self.columns = columns
        self.training = training
        self.features_type = features_type

        self.model_type = model
        if model_parameters is None:
            self.model_parameters = {}
        else:
            self.model_parameters = model_parameters
        self.sampling = sampling

        if run is None:
            self.model_folder = Path(model_folder) / self.model_parameters_string
        else:
            self.model_folder = Path(model_folder) / (self.model_parameters_string + '_run' + str(run))

        if self.model_folder.exists() is False:
            os.mkdir(self.model_folder)

        self.ground_truth = ground_truth
        self.test_areas = test_areas
        self.validation_areas = validation_areas
        self.fixed_test_area = fixed_test_area

    @property
    def model_parameters_string(self):
        parameters = self.model_type + '_' + self.features_type

        mp_str = str(self.model_parameters)
        mp_str = mp_str.replace('{', '')
        mp_str = mp_str.replace('}', '')
        mp_str = mp_str.replace("'", '')
        mp_str = mp_str.replace(':', '')
        mp_str = mp_str.replace(' ', '')

        mp_str = mp_str.replace('_', '')
        mp_str = mp_str.replace(',', '_')
        if mp_str == '':
            mp_str = 'default'

        parameters = parameters + '_' + mp_str
        return parameters

    @property
    def model(self):
        if self.model_type == 'nearest_neighbor':
            classifier = sklearn.neighbors.KNeighborsClassifier
        elif self.model_type == 'decision_tree':
            classifier = sklearn.tree.DecisionTreeClassifier
        elif self.model_type == 'gaussian_process':
            classifier = sklearn.gaussian_process.GaussianProcessClassifier
        elif self.model_type == 'svm':
            classifier = sklearn.svm.SVC
        elif self.model_type == 'rf':
            classifier = sklearn.ensemble.RandomForestClassifier
        else:
            classifier = sklearn.neural_network.MLPClassifier

        model = classifier(**self.model_parameters)
        return model

    @property
    def scaler(self):
        scaler_path = self.model_folder / 'scaler.save'

        if self.training:
            gdf_train = self.gdf[self.gdf['split'] == 'train']
            x_train = gdf_train[self.columns].to_numpy()

            scaler = sklearn.preprocessing.MinMaxScaler()
            scaler.fit(x_train)
            joblib.dump(scaler, scaler_path)
        else:
            scaler = joblib.load(scaler_path)

        return scaler

    @property
    def classifier(self):
        classifier_path = self.model_folder / 'classifier.save'

        if self.training:
            gdf_train = self.gdf[self.gdf['split'] == 'train']
            x_train = gdf_train[self.columns].to_numpy()
            y_train = gdf_train['ground_truth'].to_numpy()

            if self.sampling:
                sampling_dict = {}
                classes, counts = np.unique(y_train, return_counts=True)

                max_count = 1500
                for c in range(len(classes)):
                    c_value = classes[c]
                    c_count = counts[c]

                    if c_count < max_count:
                        sampling_dict[c_value] = c_count
                    else:
                        sampling_dict[c_value] = max_count

                under_sampler = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=sampling_dict)
                smote = imblearn.over_sampling.SMOTE(sampling_strategy='all')
                sampling_pipeline = imblearn.pipeline.Pipeline(steps=[('u', under_sampler), ('o', smote)])

                x_train, y_train = sampling_pipeline.fit_resample(x_train, y_train)

            classifier = self.model
            scaler = self.scaler
            scaled_train_data = scaler.transform(x_train)
            classifier.fit(scaled_train_data, y_train)
            joblib.dump(classifier, classifier_path)
        else:
            classifier = joblib.load(classifier_path)

        return classifier

    def corrected_area(self, test=True):
        corrected_areas = []
        if test:
            areas = self.test_areas
        else:
            areas = self.validation_areas
        for test_area in areas:
            minx, miny, maxx, maxy = test_area.bounds
            if self.fixed_test_area:
                patch_overlap = 256 / 4
            else:
                patch_overlap = 0
            minx = minx + patch_overlap
            miny = miny + patch_overlap
            maxx = maxx - patch_overlap
            maxy = maxy - patch_overlap
            poly_coords = [[minx, maxy], [maxx, maxy], [maxx, miny], [minx, miny]]
            test_area_poly = shapely.Polygon(poly_coords)
            corrected_areas.append(test_area_poly)
        test_area = shapely.unary_union(corrected_areas)
        return test_area

    def test_gdf(self, test=True):
        gdf = gpd.read_file(self.ground_truth)
        gdf['phd_class'] = gdf['phd_class'].astype(int)
        if self.fixed_test_area:
            tested_area = self.corrected_area(test=test)
            gdf = gdf.clip(tested_area, keep_geom_type=True)
        gdf = gdf.dissolve(by='phd_class')
        gdf = gdf.reset_index()
        return gdf

    def inference(self):
        scaler = self.scaler
        classifier = self.classifier

        if self.training:
            gdf_validation = self.gdf[self.gdf['split'] == 'validation']
            gdf_validation = gdf_validation.copy()
            gdf_test = self.gdf[self.gdf['split'] == 'test']
            gdf_test = gdf_test.copy()

            x_valid = gdf_validation[self.columns].to_numpy()
            x_test = gdf_test[self.columns].to_numpy()

            scaled_validation_data = scaler.transform(x_valid)
            scaled_test_data = scaler.transform(x_test)

            classified_validation = classifier.predict(scaled_validation_data)
            classified_test = classifier.predict(scaled_test_data)

            gdf_validation['prediction'] = classified_validation
            gdf_test['prediction'] = classified_test

            return gdf_validation, gdf_test

        else:
            gdf = self.gdf

            x = gdf[self.columns].to_numpy()
            scaled_x = scaler.transform(x)
            classified_x = classifier.predict(scaled_x)

            gdf['prediction'] = classified_x
            return gdf

    def per_pixel_gdf(self, gdf, test=True):
        test_area = self.corrected_area(test=test)
        test_areas = list(test_area.geoms)

        shapes_list = []
        for geom in test_areas:
            gdf_area = gdf.clip(geom, keep_geom_type=True)
            if len(gdf_area) > 0:
                minx, miny, maxx, maxy = geom.bounds
                rangex = maxx - minx
                rangey = maxy - miny
                resolution = 0.5
                width = int(rangex / resolution)
                height = int(rangey / resolution)
                out_shape = (height, width)
                transform = Affine(resolution, 0.0, minx, 0.0, -resolution, maxy)

                shapes_gdf = gdf_area[['geometry', 'prediction']]
                shapes = shapes_gdf.values.tolist()

                pred = features.rasterize(((g, v) for g, v in shapes), out_shape=out_shape, fill=0, transform=transform)

                shapes = features.shapes(pred.astype('uint8'), transform=transform)
                shapes = [(shapely.geometry.shape(s), int(v)) for s, v in shapes]

                shapes_list.append(shapes)

        shapes_list = np.concatenate(shapes_list)
        gdf_test = gpd.GeoDataFrame(shapes_list, columns=['shape', 'class'])
        gdf_test = gdf_test.set_geometry('shape', drop=True)
        gdf_test.crs = gdf.crs

        gdf_test = gdf_test.explode(index_parts=False)
        gdf_test = gdf_test[gdf_test.geometry.geom_type == 'Polygon']
        gdf_test = gdf_test.reset_index(drop=True)

        return gdf_test

    def conf_matrix_per_pixel(self, classified_gdf, test=True):
        classified_gdf = classified_gdf.dissolve(by='class')
        classified_gdf = classified_gdf.reset_index()
        gdf_test = self.test_gdf(test=test)
        cm = pd.DataFrame()

        for ac in range(8):
            actual_class = gdf_test[gdf_test['phd_class'] == ac]
            class_geom = actual_class.geometry.item()
            actual_classified = classified_gdf.clip(class_geom)
            actual_classified = actual_classified.explode(index_parts=False)
            actual_classified = actual_classified[actual_classified.geometry.geom_type == 'Polygon']
            actual_classified = actual_classified.dissolve(by='class')
            actual_classified = actual_classified.reset_index()
            actual_classified['area'] = actual_classified.area
            classified_classes = actual_classified['class'].tolist()

            for cc in range(8):
                if cc in classified_classes:
                    ac_cc_gdf = actual_classified[actual_classified['class'] == cc]
                    ac_cc_area = ac_cc_gdf['area'].item()
                else:
                    ac_cc_area = 0

                cm.loc[ac, cc] = int(ac_cc_area * 4)
        cm = cm.astype(int)
        # cm.to_csv(self.model_folder / 'confusion_matrix_per_pixel.csv')
        return cm

    @staticmethod
    def conf_matrix_per_object(classified_gdf):
        y_true = classified_gdf['ground_truth'].to_numpy()
        y_pred = classified_gdf['prediction'].to_numpy()

        cm_numpy = sklearn.metrics.confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(data=cm_numpy)
        return cm_df

    @staticmethod
    def accuracy(cm):
        tp_fn_class = cm.values.sum(axis=1)
        tp_fp_class = cm.values.sum(axis=0)

        tp_class = np.array([cm.values[c, c] for c in range(8)])
        fn_class = tp_fn_class - tp_class
        fp_class = tp_fp_class - tp_class

        oa = tp_class.sum() / cm.values.sum()
        pa = tp_class / tp_fn_class
        pa[tp_fn_class == 0] = 0
        ua = [tp_class[c] / tp_fp_class[c] if tp_fp_class[c] != 0 else 0 for c in range(len(tp_class))]
        ji_class = tp_class / (tp_class + fn_class + fp_class)

        acc_df = pd.DataFrame()
        acc_df['overall accuracy'] = [oa]
        for c in range(8):
            acc_df['producers_accuracy_class' + str(c)] = [pa[c]]
            acc_df['users_accuracy_class' + str(c)] = [ua[c]]
        acc_df['mean iou'] = [np.mean(ji_class)]
        acc_df['mean iou without other and forest'] = [np.mean(ji_class[1:-1])]

        for c in range(8):
            acc_df['ji_class' + str(c)] = [ji_class[c]]

        return acc_df

    def test(self):
        if self.training:
            gdf_validation_object, gdf_test_object = self.inference()
            gdf_validation_pixel = self.per_pixel_gdf(gdf_validation_object, test=False)
            gdf_test_pixel = self.per_pixel_gdf(gdf_test_object)

            gdf_validation_object.to_file(self.model_folder / 'validation_object.gpkg')
            gdf_test_object.to_file(self.model_folder / 'test_object.gpkg')
            gdf_validation_pixel.to_file(self.model_folder / 'validation_pixel.gpkg')
            gdf_test_pixel.to_file(self.model_folder / 'test_pixel.gpkg')

            cm_validation_object = self.conf_matrix_per_object(gdf_validation_object)
            cm_test_object = self.conf_matrix_per_object(gdf_test_object)
            cm_validation_pixel = self.conf_matrix_per_pixel(gdf_validation_pixel, test=False)
            cm_test_pixel = self.conf_matrix_per_pixel(gdf_test_pixel)

            cm_validation_object.to_csv(self.model_folder / 'cm_validation_object.csv')
            cm_test_object.to_csv(self.model_folder / 'cm_test_object.csv')
            cm_validation_pixel.to_csv(self.model_folder / 'cm_validation_pixel.csv')
            cm_test_pixel.to_csv(self.model_folder / 'cm_test_pixel.csv')

            acc_validation_object = self.accuracy(cm_validation_object)
            acc_test_object = self.accuracy(cm_test_object)
            acc_validation_pixel = self.accuracy(cm_validation_pixel)
            acc_test_pixel = self.accuracy(cm_test_pixel)

            acc_validation_object.to_csv(self.model_folder / 'acc_validation_object.csv')
            acc_test_object.to_csv(self.model_folder / 'acc_test_object.csv')
            acc_validation_pixel.to_csv(self.model_folder / 'acc_validation_pixel.csv')
            acc_test_pixel.to_csv(self.model_folder / 'acc_test_pixel.csv')

        else:
            print('Testing can only be performed when training the model.')
