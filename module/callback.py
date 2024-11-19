import dataclasses
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
from tqdm.auto import tqdm

from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import TrainingArguments
from transformers.utils import logging
from transformers import TrainerCallback, TrainerState, TrainerControl
from utils.utils import ModelState

logger = logging.get_logger(__name__)


class ProcessTextEmbeddingCallback(TrainerCallback):

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if kwargs['model'].lm_state == ModelState.ON:
            kwargs['model'].text_embedding = None
        else:
            kwargs['model'].build_text_embedding()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if control.should_evaluate:
            kwargs['model'].build_text_embedding()

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if control.should_evaluate:
            kwargs['model'].build_text_embedding()


class AlterCallback(TrainerCallback):
    """
    Alternatively switch a part of model to froze
    different part has its own logging/evaluation strategy, learning rate, optimizer, scheduler
    """

    def __init__(self, 
                 lm_universal_strategy : str = 'steps', lm_universal_interval : int = 500, 
                 lm_alter_strategy : str = 'epoch', lm_alter_interval : int = 1, 
                 ctr_universal_strategy : str = 'steps', ctr_universal_interval : int = 500, 
                 ctr_alter_strategy : str = 'epoch', ctr_alter_interval : int = 1,
                 **kwargs
                ):
        assert lm_alter_strategy == 'epoch'
        assert ctr_alter_strategy == 'epoch'
        
        # Language model
        self.lm_universal_strategy = lm_universal_strategy
        self.lm_universal_interval = lm_universal_interval
        self.lm_univeral_counter = 0

        self.lm_alter_strategy = lm_alter_strategy
        self.lm_alter_interval = lm_alter_interval
        self.lm_alter_counter = 0

        # if lm_alter_interval % lm_universal_interval != 0:
        #     raise ValueError('lm_alter_interval should mod lm_universal_interval.')
        
        # CTR model
        self.ctr_universal_strategy = ctr_universal_strategy
        self.ctr_universal_interval = ctr_universal_interval
        self.ctr_univeral_counter = 0

        self.ctr_alter_strategy = ctr_alter_strategy
        self.ctr_alter_interval = ctr_alter_interval
        self.ctr_alter_counter = 0

        # if ctr_alter_interval % ctr_universal_interval != 0:
        #     raise ValueError('ctr_alter_interval should mod ctr_universal_interval.')

    
    def on_train_begin(self, args, state, control, **kwargs):
        kwargs['model'].freeze('ctr_model')


    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs['model']
        if model.lm_state == ModelState.ON:
            assert model.ctr_state == ModelState.OFF, 'lm_state should be different from ctr_state.'
            state.training_model = 'language_model'
        else:
            assert model.lm_state == ModelState.OFF, 'ctr_state should be different from lm_state.'
            state.training_model = 'ctr_model'
        
        
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        if model.lm_state == ModelState.ON:
            assert model.ctr_state == ModelState.OFF, 'lm_state should be different from ctr_state.'

            # Log / Eval / Save
            if self.lm_universal_strategy == IntervalStrategy.STEPS:
                self.lm_univeral_counter += 1
            if self.lm_universal_strategy == IntervalStrategy.STEPS and self.lm_univeral_counter % self.lm_universal_interval == 0:
                control.should_log = True
                control.should_evaluate = True
                control.should_save = True
            else:
                control.should_log = False
                control.should_evaluate = False
                control.should_save = False

            # # Alter
            # if self.lm_alter_strategy == IntervalStrategy.STEPS:
            #     self.lm_alter_counter += 1
            # if self.lm_alter_strategy == IntervalStrategy.STEPS and self.lm_alter_counter % self.lm_alter_interval == 0:
            #     model.switch_states()

        elif model.ctr_state == ModelState.ON:         
            assert model.lm_state == ModelState.OFF, 'ctr_state should be different from lm_state.'

            # Log / Eval / Save
            if self.ctr_universal_strategy == IntervalStrategy.STEPS:
                self.ctr_univeral_counter += 1
            if self.ctr_universal_strategy == IntervalStrategy.STEPS and self.ctr_univeral_counter % self.ctr_universal_interval == 0:
                control.should_log = True
                control.should_evaluate = True
                control.should_save = True
            else:
                control.should_log = False
                control.should_evaluate = False
                control.should_save = False

            # # Alter
            # if self.ctr_alter_strategy == IntervalStrategy.STEPS:
            #     self.ctr_alter_counter += 1
            # if self.ctr_alter_strategy == IntervalStrategy.STEPS and self.ctr_alter_counter % self.ctr_alter_interval == 0:
            #     model.switch_states()

        if state.global_step >= state.max_steps:
            control.should_training_stop = True

        return control
    


    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        if model.lm_state == ModelState.ON:
            assert model.ctr_state == ModelState.OFF, 'lm_state should be different from ctr_state.'

            # Log / Eval / Save
            if self.lm_universal_strategy == IntervalStrategy.EPOCH:
                self.lm_univeral_counter += 1
            if self.lm_universal_strategy == IntervalStrategy.EPOCH and self.lm_univeral_counter % self.lm_universal_interval == 0:
                control.should_log = True
                control.should_evaluate = True
                control.should_save = True
            else:
                control.should_log = False
                control.should_evaluate = False
                control.should_save = False

            # Alter
            if self.lm_alter_strategy == IntervalStrategy.EPOCH:
                self.lm_alter_counter += 1
            if self.lm_alter_strategy == IntervalStrategy.EPOCH and self.lm_alter_counter % self.lm_alter_interval == 0:
                model.switch_states()

        elif model.ctr_state == ModelState.ON:
            assert model.lm_state == ModelState.OFF, 'ctr_state should be different from lm_state.'
            
            # Log / Eval / Save
            if self.ctr_universal_strategy == IntervalStrategy.EPOCH:
                self.ctr_univeral_counter += 1
            if self.ctr_universal_strategy == IntervalStrategy.EPOCH and self.ctr_univeral_counter % self.ctr_universal_interval == 0:
                control.should_log = True
                control.should_evaluate = True
                control.should_save = True
            else:
                control.should_log = False
                control.should_evaluate = False
                control.should_save = False

            # Alter
            if self.ctr_alter_strategy == IntervalStrategy.EPOCH:
                self.ctr_alter_counter += 1
            if self.ctr_alter_strategy == IntervalStrategy.EPOCH and self.ctr_alter_counter % self.ctr_alter_interval == 0:
                model.switch_states()

        return control
    

    # def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     model = kwargs['model']
    #     if model.lm_state == ModelState.ON:
    #         state.training_model = 'language_model'
    #     elif model.ctr_state == ModelState.ON:
    #         state.training_model = 'ctr_model'