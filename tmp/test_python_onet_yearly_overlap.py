"""Small deterministic end-to-end checks for posting weighting and cache joins."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT = Path(__file__).with_name("python_onet_yearly_overlap.py")
spec = importlib.util.spec_from_file_location("python_onet_overlap", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class PostingOverlapTest(unittest.TestCase):
    def fixture(self, root):
        cache = root / "cache"
        (cache / "layer_12").mkdir(parents=True)
        predictions = root / "predictions"
        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"num_hidden_layers":12}))
        metadata, cache_index = [], []
        # In 2010 A1 has three occurrences; A2 has one. They still have equal
        # posting weight. Both source files reuse row_idx=0.
        year_inputs = {
            2010:[("A1", "15-1252.00", [8,4,0,0,0,0], 3),
                  ("A2", "15-1252.00", [0,0,0,6,2,0], 1),
                  ("B1", "15-1254.00", [1,0,0,0,0,0], 1)],
            2011:[("A3", "15-1252.00", [8,0,4,0,0,0], 1),
                  ("A4", "15-1252.00", [0,0,0,2,6,0], 1),
                  ("B2", "15-1254.00", [1,0,0,0,0,0], 1)],
        }
        next_id = 0
        encoded_counts = {}
        for year, postings in year_inputs.items():
            directory = predictions / str(year)
            directory.mkdir(parents=True)
            path = directory / f"predictions_{year}-01.csv.gz"
            prediction_rows, features, ids = [], [], []
            prediction_rows.append({"true_skill":"java", "masked_sentence":"a [MASK] job",
                                    "row_idx":0,"file_path":"/raw/unselected.csv.gz", "onet_name":"Other"})
            for pid, code, vector, occurrences in postings:
                name = "Software Developers" if code == "15-1252.00" else "Web Developers"
                source = f"/raw/{pid}.csv.gz"
                metadata.append({"id":pid,"posted":f"{year}-01-07", "posted_year":year,
                                 "onet":code,"onet_name":name,"source_file":source,"row_idx":0,
                                 "n_python_instances":occurrences,"n_valid_mask_instances":occurrences})
                for occurrence in range(occurrences):
                    prediction_rows.append({"true_skill":"Python", "masked_sentence":f"{pid} use [MASK] {occurrence}",
                                            "row_idx":0,"file_path":source,"onet_name":name})
                    cache_index.append({"sample_id":next_id,"source_row_idx":"0",
                                        "source_file":path.name,"year":year})
                    features.append(vector)
                    ids.append(next_id)
                    next_id += 1
            pd.DataFrame(prediction_rows).to_csv(path,index=False,compression="gzip")
            array = np.asarray(features,dtype=np.float32)
            table = pa.table({"sample_id":pa.array(ids,type=pa.int64()),
                              "year":pa.array([year]*len(ids),type=pa.int16()),
                              "layer":pa.array([12]*len(ids),type=pa.int16()),
                              "features":pa.FixedSizeListArray.from_arrays(pa.array(array.ravel()),6)})
            pq.write_table(table,cache/"layer_12"/f"year_{year}.parquet")
            encoded_counts[str(year)] = len(ids)
        metadata_path = root / "metadata.csv.gz"
        pd.DataFrame(metadata).to_csv(metadata_path,index=False,compression="gzip")
        pd.DataFrame(cache_index).to_parquet(cache/"data_index.parquet",index=False)
        manifest = {"complete":True,"feature_dim":6,"skipped_no_mask":0,
                    "eligible_counts_by_year":encoded_counts,"selected_counts_by_year":encoded_counts,
                    "encoded_counts_by_year":encoded_counts,
                    "signature":{"format_version":1,"target_skill":"python","data_mode":"all",
                                 "layers":[12],"years":[2010,2011],"predictions_root":str(predictions),
                                 "model_name":str(model),"truth_col":"true_skill",
                                 "text_col":"masked_sentence","row_id_col":"row_idx"}}
        (cache/"manifest.json").write_text(json.dumps(manifest))
        return cache, predictions, metadata_path

    def run_cli(self, root, cache, predictions, metadata, stage="all"):
        argv = [str(SCRIPT),"--cache-dir",str(cache),"--predictions-root",str(predictions),
                "--posting-metadata",str(metadata),"--output-dir",str(root/"output"),
                "--years","2010-2011","--min-postings-per-year","2","--expected-occupations","1",
                "--topk","2","--dpi","40","--figure-size","4","--feature-batch-size","2",
                "--read-chunksize","2","--stage",stage]
        with patch("sys.argv",argv), contextlib.redirect_stdout(io.StringIO()):
            analysis.main()

    def test_equal_posting_weight_and_legacy_feature_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache,predictions,metadata = self.fixture(root)
            self.run_cli(root,cache,predictions,metadata)
            output = root/"output"
            selected = pd.read_csv(output/"selected_occupations.csv")
            self.assertEqual(selected["onet_code"].tolist(),["15-1252.00"])
            with np.load(output/"group_means.npz") as data:
                np.testing.assert_allclose(data["means"][0,0],[4,2,0,3,1,0])
                np.testing.assert_allclose(data["means"][0,1],[4,0,2,1,3,0])
            vectors = pq.read_table(output/"posting_representations/year_2010.parquet").to_pydict()
            self.assertEqual(vectors["posting_id"],["A1","A2"])
            self.assertEqual(vectors["n_python_instances"],[3,1])
            np.testing.assert_allclose(vectors["features"],[[8,4,0,0,0,0],[0,0,0,6,2,0]])
            matrix = pd.read_csv(output/"15-1252.00/overlap_matrix_layer_12.csv",index_col=0)
            np.testing.assert_array_equal(matrix.to_numpy(),[[2,0],[0,2]])
            removed = json.loads((output/"15-1252.00/excluded_feature_ids.json").read_text())
            self.assertEqual(removed["excluded_feature_ids"],[0])
            run = json.loads((output/"run_manifest.json").read_text())
            self.assertEqual(run["n_selected_postings"],4)
            self.assertEqual(run["n_selected_instances"],6)
            self.assertEqual(run["status"],"complete")
            self.assertTrue((output/"15-1252.00/overlap_heatmap_large.png").is_file())
            self.run_cli(root,cache,predictions,metadata,stage="plot")

    def test_rejects_misaligned_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache,predictions,metadata = self.fixture(root)
            index = pd.read_parquet(cache/"data_index.parquet")
            index.loc[0,"source_row_idx"] = "99"
            index.to_parquet(cache/"data_index.parquet",index=False)
            with self.assertRaisesRegex(ValueError,"Cache row order mismatch"):
                self.run_cli(root,cache,predictions,metadata)

    def test_rejects_incomplete_audit_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache,predictions,metadata = self.fixture(root)
            frame = pd.read_csv(metadata)
            frame.loc[0,["n_python_instances","n_valid_mask_instances"]] = 4
            frame.to_csv(metadata,index=False,compression="gzip")
            with self.assertRaisesRegex(ValueError,"complete test_clean audit"):
                self.run_cli(root,cache,predictions,metadata)


if __name__ == "__main__":
    unittest.main()
