# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")


def train():
  """Train CIFAR-10 for a number of steps."""

  """
  TensorFlow에서 연산은 데이터 흐름을 보여주는 Graph로 표현된다
  Graph에는 연산 유닛으로 표현되는 일련의 tf.Operation 객체들과 operation들 사이에 오가는 데이터 유닛으로 표현되는 tf.Tensor 객체들을 포함한다.
  tf.get_default_graph를 호출함으로써 기본 Graph를 등록하고 접근할 수 있다. 기본 Graph에 operation을 추가하기 위해서는 간단하게 새로운
  Operation에 정의된 함수 하나를 호출하면 된다
  
  또다른 전형적인 방법은 context의 생명주기 내에서 현재의 기본 Graph를 재정의한 tf.Graph.as_default라는 context manager를 아용하는 것이다
  
  주의할 것은 Graph 클래스는 graph 생성과 관련하여 thread로부터 안전하지 않기 때문에 모든 operation들은 단일 thread에서 생성하거나 외부
  동기화를 제공해야 한다. 또한 별도로 명시되지 않는 한 모든 메소드역시 thread로부터 안전하지 않다.
  
  graph 인스턴스는 이름으로 식별되는 임의의 수의 collection을 지원한다. 규모가 큰 graph를 만들 때 이 collection에는 관련된 객체들의 그룹을
  저장할 수 있다. 예를들면 tf.Variable은 graph가 생성되는 동안 만들어지는 모든 변수들을 저장하기 위해 (tf.GraphKeys.GLOBAL_VARIABLES라는
  이름의)collection을 사용한다.    
  """
  with tf.Graph().as_default():
      # 인자로 주어진 graph내의 global step tensor를 반환(존재하지 않으면 생성하여 반환)한다. 인자가 주어지지 않으면 default graph의
      # global step을 반환한다.
    global_step = tf.train.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on`
    # GPU and resulting in a slow down.

      # 컴퓨터의 첫 번째 cpu를 이용하여 cifar10 클래스로부터 변형된 입력 이미지들과 해당 라벨을 받아온다.
    with tf.device('/cpu:0'):
      images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.

      # 예측 모델을 만든다.
    logits = cifar10.inference(images)

    # Calculate loss.

      # 손실값을 계산한다.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.

      # 학습 모델을 가져온다.
    train_op = cifar10.train(loss, global_step)

      # tf.train.SessionRunHook를 상속받아 _LoggerHook라는 로그 관리 클래스를 만든다
      # MonitoredSession.run()의 프로세스를 가로채 로그 처리를 할 때 사용한다
    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

        # session을 사용하기 전에 처음 한 번만 호출된다.
        # 여기서는 전역 변수로 _step 값과 _start_time 값을 초기화 한다
      def begin(self):
        self._step = -1
        self._start_time = time.time()

        # session.run()이 호출되기 전에 호출된다.
        # _step 값을 하나 증가시키고 tf.train.SessionRunArgs(loss)를 리턴하는데 tf.train.SessionRunArgs(loss)는
        # session.run()에 추가되어야 할 인자로 loss 값을 전달하는 것이다.
        # run_context 인자는 SessionRunContext 클래스의 인스턴스로 다음 Session run에 사용될 정보를 제공한다.
      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

        # session.run()이 호출된 후에 호출된다. run_context 인자는 before_run의 run_context와 동일한 인자이며
        # run_values 인자는 before_run에서 요청된 작업이나 텐서를 포함한다
        # 스크립트 상단에 FLAGS.log_frequency가 10으로 설정되어있으므로 10의 배수 번째 스텝마다 내용을 print한다.
      def after_run(self, run_context, run_values):
          # 10의 배수 번째 step마다...
        if self._step % FLAGS.log_frequency == 0:
            # 현재 시간을 가져와서...
          current_time = time.time()
            # 경과 시간을 계산한다.
          duration = current_time - self._start_time
            # 다음 step의 경과 시간 계산을 위해 _start_time에 현재 시간을 넣는다
          self._start_time = current_time

            # loss_value에 before_run에서 반환한 tf.train.SessionRunArgs(loss) 값을 담는다
          loss_value = run_values.results
            # step이 경과된 시간 동안 학습된 이미지 숫자를 계산한다
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
            # 배치당 수행된 시간을 계산한다
          sec_per_batch = float(duration / FLAGS.log_frequency)

            # 출력할 로그 포맷을 설정한다
          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
            # 로그를 출력한다.
            # 로그 형태는 다음과 같다.
            # 2018-05-09 15:15:54.447879: step 100, loss = 4.07 (394.2 examples/sec; 0.325 sec/batch)
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

        # session을 만든다. tf.train.MonitoredTrainingSession는 학습을 위한 MonitoredSession을 생성시켜주는 유틸리티 클래스이다.
        # MonitoredSession은 분산 환경에서 초기화, 복구, hook 등의 기능을 제공하는 Session 형태의 객체이다.
        # 분산 환경에서 TensorFlow는 Worker를 통해 연산을 수행하는데 MonitoredSession은 이 중 중요(chief) Worker에
        # 적절한 세션 초기화와 복구 기능을 설정해주며 체크포인트 및 summary 저장과 관련된 hook를 생성해준다. 일반 Worker들의 측면에서는
        # 중요(chief) Worker가 초기화 혹은 복구 진행 중일 때 대기할 수 있도록 적절한 Session 생성자를 설정해준다.
        #
        # 아래 사용된 인자들은 다음과 같은 의미가 있다.
        # . checkpoint_dir : 복구될 변수들이 저장될 디렉토리.
        # . hooks : SessionRunHook의 객체들의 list로 현재 전달된 객체들은 다음과 같다.
        #           - tf.train.StopAtStepHook : 특정 step에서 중단 요청을 보내는 hook.
        #                                       여런 step이 실행된 후나 마지막 step이 실행된 후에 중지 요청을 하게 되면 이 둘 중
        #                                       하나의 방법만 사용할 수 있다.
        #           - tf.train.NanTensorHook : loss를 감시하고 있다가 loss 값이 NaN이 되는 경우 중지를 요청하는 hook.
        #           - _LoggerHook() : 로그 출력을 위해 바로 위에서 선언한 클래스.
        # . config : tf.ConfigProto 타입의 인자로 tf.Session의 인자로 사용되는 프로토콜 버.
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
        # 위에서 선언한 _LoggerHook 클래스 내의 after_run의 인자인 run_context는 SessionRunContext 클래스의 인스턴스로
        # 필요한 경우 session의 반복 처리를 중지하기 위해 request_stop()을 호출할 수 있다. 특히 분산 환경에서 Worker 중 하나가
        # request_stop()을 호출하게 되면 should_stop()이 True를 리턴하게 되기 때문에 should_stop()를 체크하고 있는 모든
        # Worker의 loop가 모두 종료된다.
      while not mon_sess.should_stop():
          # session을 통해 모델 학습을 시작
        mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
    # CIFAR-10 데이터를 다운로드 받아 로컬 디렉토리에 저장함
  #cifar10.maybe_download_and_extract()

    # 학습 데이터가 저장된 경로가 존재하는지 확인
  if tf.gfile.Exists(FLAGS.train_dir):
      # 학습 데이터 경로가 존재한다면 하위 디렉토리를 포함하여 모두 삭제
    tf.gfile.DeleteRecursively(FLAGS.train_dir)

    # 새로 학습 데이터 경로를 만들고
  tf.gfile.MakeDirs(FLAGS.train_dir)
    # 학습 시작
  train()

# 이 스크립트 파일을 직접 실행시킬 경우에만 tf.app.run()이 실행됨
# 이 스크립트 파일이 다른 스크립트에 import 되는 경우 if문의 실행문(tf.app.run())은 실행되지 않음
if __name__ == '__main__':
  # 파라미터로 전달된 인자를 넘겨 main 함수를 실행함. 파라미터가 없으니 기본 main 함수를 실행함
  tf.app.run()
