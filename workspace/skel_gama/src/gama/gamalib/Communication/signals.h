/*
 * signals.h
 *
 *  Created on: Apr 3, 2012
 *      Author: jbarbosa
 */

#ifndef SIGNALS_H_
#define SIGNALS_H_


/*!
 * @brief Device Synchronization flags
 * @note DO NOT CHANGE
 */
enum SIGNAL_DEVICE {
	NULL_SIGNAL = 0,  /*!<NULL Signal */
	START_SIGNAL,  /*!< Start location execution */
	SYNC_SIGNAL,  /*!< Synchronize with all devices */
	SYNCED_SIGNAL,  /*!< Synchronize with all devices */
	TERMINATE_SIGNAL,  /*!< Terminate local execution */
	TERMINATE_SIGNAL_CHILD  /*!< Terminate local execution */

};

/*!
 * @brief Device Status
 * @note DO NOT CHANGE
 */
enum DEVICE_STATUS {
	DEVICE_NOT_INITIALIZED=0, /*!< Device not initialized */
	DEVICE_INITIALIZED, /*!< Device initialized */
	DEVICE_KERNEL_RUNNING, /*!< Device Kernel running */
	DEVICE_KERNEL_SYNCED, /*!< Device Kernel synced */
	DEVICE_KERNEL_TERMINATED, /*!< Device Kernel terminated */
	DEVICE_NULL_STATUS = 0xff
};
/*!
 * @brief DEVICE Communication Streams
 * @note DO NOT CHANGE
 */
enum DEVICE_STREAMS {
	DEVICE_INPUT_STREAM=0, /*!< Device Input stream */
	DEVICE_OUTPUT_STREAM, /*!< Device Output stream */
	DEVICE_KERNEL_STREAM, /*!< Device Kernel stream */
	DEVICE_TOTAL_STREAMS /*!< Total number of streams */
};

/*!
 * @brief DEVICE Communication Flags
 * @note DO NOT CHANGE
 */

enum COMMUNICATION_FLAGS {
	CPU_CONTROL=0,  /*!< CPU as control */
	DEVICE_CONTROL,  /*!< Device as control */
	DEVICE_CPU_SYNC,  /*!< Sync Device and CPU */
	TERMINATE_KERNEL  /*!< Terminate current kernel */
};
/*!
 * @brief DEVICE Communication Channel
 * @note DO NOT CHANGE
 */
enum COMMUNICATION_GATES {
	DEVICE_IN=0, /*!< Device data in*/
	DEVICE_OUT,  /*!< Device data out */
	DEVICE_KERNEL,  /*!< Device Kernel */
	TOTAL_CHANNELS  /*!< Total channels */
};

/*!
 * @brief DEVICE copy direction
 * @note DO NOT CHANGE
 */
typedef enum COPY_DIRECTION {
	HOST_TO_HOST=0, /*!< Copy host to host */
	HOST_TO_DEVICE, /*!< Copy host to DEVICE*/
	DEVICE_TO_HOST /*!< Copy DEVICE to HOST*/
} CopyDirection;


#endif /* SIGNALS_H_ */
